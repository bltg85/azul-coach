import copy
import random
import unittest
from unittest.mock import patch

import numpy as np

import _framework_path
from model import GameState, GameRunner
from agents.heuristic import HeuristicPlayer
from agents.sim import AzulSim
from agents.mcts import MCTSPlayer
from agents.opponent_search import OpponentSearchPlayer, MovePredictor, opponent_distribution, public_key
from az.actions import move_to_action
from az.net import NumpyNet
from az.player import AZPlayer


def position(seed=7):
    gs = GameState(4, rng=random.Random(seed))
    for p in gs.players:
        p.player_trace.StartRound()
    return AzulSim(gs, gs.first_player)


class SearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.net = NumpyNet.load("az/weights/az_v1.npz")

    def test_clones_do_not_share_rng_or_mutable_state(self):
        sim = position()
        clone = sim.clone()
        before = copy.deepcopy(sim.gs)
        clone.gs.rng.random()
        clone.apply(clone.legal_moves()[0])
        self.assertEqual(sim.gs.rng.getstate(), before.rng.getstate())
        self.assertEqual(sim.gs.bag, before.bag)
        for a, b in zip(sim.gs.players, before.players):
            self.assertEqual(a.lines_number, b.lines_number)
            np.testing.assert_array_equal(a.grid_state, b.grid_state)

    def test_draws_are_independent_of_global_random(self):
        a, b = position(), position()
        a.gs.bag_used = a.gs.bag + a.gs.bag_used
        b.gs.bag_used = b.gs.bag + b.gs.bag_used
        a.gs.bag = []; b.gs.bag = []
        a.gs.SetupNewRound()
        for _ in range(1000):
            random.random()
        b.gs.SetupNewRound()
        self.assertEqual([f.tiles for f in a.gs.factories], [f.tiles for f in b.gs.factories])

    def test_hidden_sampling_ignores_order_and_real_rng(self):
        a = position()
        b = a.clone()
        b.gs.bag.reverse()
        b.gs.rng.seed(999)
        x = a.sample_hidden(random.Random(41))
        y = b.sample_hidden(random.Random(41))
        self.assertEqual(x.gs.bag, y.gs.bag)
        self.assertEqual(x.gs.rng.getstate(), y.gs.rng.getstate())
        self.assertEqual(sorted(x.gs.bag), sorted(a.gs.bag))
        self.assertEqual(public_key(a), public_key(b))
        z = a.sample_hidden(random.Random(42))
        self.assertNotEqual(x.gs.bag, z.gs.bag)

    def test_chance_layouts_and_discard_counts_have_different_keys(self):
        a = position(); b = a.clone()
        tile = b.gs.bag.pop()
        b.gs.bag_used.append(tile)
        self.assertNotEqual(public_key(a), public_key(b))
        b = a.clone()
        b.gs.factories[0].tiles[0] += 1
        self.assertNotEqual(public_key(a), public_key(b))

    def test_tie_break_and_shared_wins(self):
        sim = position()
        for p in sim.gs.players:
            p.score = 50
        sim.gs.players[1].grid_state[0, :] = 1
        sim.terminal = True
        self.assertEqual(sim.winners(), [1])
        self.assertEqual(sim.win_values(), [0, 1, 0, 0])
        sim.gs.players[3].grid_state[0, :] = 1
        self.assertEqual(sim.win_values(), [0, .5, 0, .5])

    def test_no_win_target_for_unfinished_game(self):
        with self.assertRaises(ValueError):
            position().win_values()

    def test_probability_tail_preserves_unexpected_moves(self):
        p = np.array([.6, .2, .1, .06, .04])
        q = opponent_distribution(p)
        self.assertAlmostEqual(float(q.sum()), 1)
        self.assertTrue(np.all(q > 0))
        self.assertGreater(q[:3].sum(), p[:3].sum())
        np.testing.assert_allclose(opponent_distribution(p, tail_weight=1), p)

    def test_prediction_uses_new_position_after_a_move(self):
        sim = position()
        predictor = MovePredictor(self.net)
        _, mapping, _, _ = predictor.predict(sim.gs, sim.cur)
        move = next(m for m in mapping.values() if m[1] == 0)
        sim.apply(move)
        _, after, _, _ = predictor.predict(sim.gs, sim.cur)
        self.assertTrue(all(m[1] != 0 for m in after.values()))

    def test_all_served_searches_ignore_hidden_order_and_global_rng(self):
        for factory in (
            lambda pid: OpponentSearchPlayer(pid, self.net, iterations=6, seed=5),
            lambda pid: AZPlayer(pid, self.net, n_sims=8, seed=5, algo="gumbel"),
            lambda pid: MCTSPlayer(pid, iterations=6, seed=5),
        ):
            a = position(); b = a.clone()
            b.gs.bag.reverse(); b.gs.rng.seed(99)
            before = public_key(a), a.gs.rng.getstate(), a.gs.bag[:]
            global_before = random.getstate()
            x = factory(a.cur).SelectMove(a.legal_moves(), a.gs)
            y = factory(b.cur).SelectMove(b.legal_moves(), b.gs)
            self.assertEqual(move_to_action(x), move_to_action(y))
            self.assertEqual(random.getstate(), global_before)
            self.assertEqual((public_key(a), a.gs.rng.getstate(), a.gs.bag), before)

    def test_search_samples_each_simulation_and_accounts_visits(self):
        sim = position()
        bot = OpponentSearchPlayer(sim.cur, self.net, iterations=7, seed=4)
        samples = []
        original = AzulSim.sample_hidden
        def sampled(sim, rng):
            result = original(sim, rng)
            samples.append(tuple(result.gs.bag))
            return result
        with patch.object(AzulSim, "sample_hidden", sampled):
            bot.SelectMove(sim.legal_moves(), sim.gs)
        self.assertEqual(len(samples), 7)
        self.assertGreater(len(set(samples)), 1)
        self.assertEqual(bot.last_stats["root_visits"], 7)
        self.assertEqual(sum(c["visits"] for c in bot.last_stats["candidates"]), 7)
        self.assertTrue(all(0 <= c["avg_value"] <= 1 for c in bot.last_stats["candidates"]))

    def test_tiny_time_budget_still_returns_a_legal_move(self):
        sim = position()
        bot = OpponentSearchPlayer(sim.cur, self.net, iterations=10,
                                   seed=0, time_budget_s=1e-9)
        bot.SelectMove(sim.legal_moves(), sim.gs)
        self.assertEqual(bot.last_stats["iterations"], 1)

    def test_simulator_matches_framework_for_two_three_and_four_players(self):
        for n in (2, 3, 4):
            bots = [HeuristicPlayer(i) for i in range(n)]
            runner = GameRunner(bots, seed=17)
            gs = copy.deepcopy(runner.game_state)
            for p in gs.players:
                p.player_trace.StartRound()
            sim = AzulSim(gs, gs.first_player)
            for _ in range(400):
                if sim.terminal:
                    break
                sim.apply(bots[sim.cur].SelectMove(sim.legal_moves(), sim.gs))
                # Every tile is on a board, in a display, or in one of the bags.
                count = len(sim.gs.bag) + len(sim.gs.bag_used)
                count += sum(f.total for f in sim.gs.factories) + sim.gs.centre_pool.total
                count += sum(sum(p.lines_number) + len(p.floor_tiles) + int(p.grid_state.sum()) for p in sim.gs.players)
                self.assertEqual(count, 100)
            self.assertTrue(sim.terminal)
            result = runner.Run(False)
            self.assertEqual(sim.scores(), [result[i][0] for i in range(n)])


if __name__ == "__main__":
    unittest.main()
