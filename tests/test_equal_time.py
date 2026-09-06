import itertools
import random
import unittest
from unittest.mock import patch

import numpy as np
import _framework_path
from model import GameState
from agents.sim import AzulSim
from agents.opponent_search import public_key
from az.actions import move_to_action
from az.net import NumpyNet
from az.timed_player import TimedAZPlayer
from bench_equal_time import summarize


class EqualTimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.net = NumpyNet.load("az/weights/az_v1.npz")

    def test_timed_search_ignores_hidden_order_and_global_rng(self):
        gs = GameState(4, rng=random.Random(42))
        sim = AzulSim(gs, gs.first_player)
        other = sim.clone()
        other.gs.bag.reverse()
        other.gs.rng.seed(9)
        before = public_key(sim), gs.bag[:], gs.rng.getstate(), random.getstate()
        selected = []
        for world in (sim, other):
            bot = TimedAZPlayer(sim.cur, self.net, .05, seed=7)
            with patch("az.timed_player.time.perf_counter", side_effect=itertools.count(0, .0001)):
                move = bot.SelectMove(world.legal_moves(), world.gs)
            selected.append(move_to_action(move))
            self.assertGreater(bot.last_stats["iterations"], 0)
        self.assertEqual(selected[0], selected[1])
        self.assertEqual(before, (public_key(sim), gs.bag, gs.rng.getstate(), random.getstate()))

    def test_non_power_of_two_candidates_use_full_budget(self):
        gs = GameState(4, rng=random.Random(42))
        moves = gs.players[gs.first_player].GetAvailableMoves(gs)
        for candidates in (3, 5, 7, 16):
            bot = TimedAZPlayer(gs.first_player, self.net, .05, seed=7)
            with patch("az.timed_player.MAX_CONSIDERED", candidates), patch(
                    "az.timed_player.time.perf_counter", side_effect=itertools.count(0, .0001)):
                bot.SelectMove(moves, gs)
            self.assertGreaterEqual(bot.last_stats["elapsed_s"], .05)
            self.assertLess(bot.last_stats["elapsed_s"], .052)

    def test_expired_setup_budget_returns_legal_prior_move_without_simulating(self):
        gs = GameState(4, rng=random.Random(42))
        bot = TimedAZPlayer(gs.first_player, self.net, .001, seed=7)
        with patch("az.timed_player.time.perf_counter", side_effect=itertools.count(0, 1)):
            move = bot.SelectMove(gs.players[gs.first_player].GetAvailableMoves(gs), gs)
        self.assertEqual(bot.last_stats["iterations"], 0)
        self.assertIsNotNone(move)

    def test_summary_clusters_seats_and_separates_both_bot_timings(self):
        records = [{"budget_s": .1, "deal_seed": seed, "subject_seat": seat,
                    "winners": [seat] if seed == 1 else [(seat+1)%4],
                    "win_share": float(seed == 1),
                    "moves": [{"seat": seat, "seconds": .11, "simulations": 5},
                              {"seat": (seat+1)%4, "seconds": .1, "simulations": 100}]}
                   for seed in (1, 2) for seat in range(4)]
        report = summarize(records)["0.1"]
        self.assertEqual(report["win_share"], .5)
        self.assertEqual(report["seed_cluster_bootstrap_95pct"], [0, 1])
        self.assertAlmostEqual(report["timings"]["learned"]["mean_s"], .11)
        self.assertAlmostEqual(report["timings"]["az"]["mean_s"], .1)
        incomplete = summarize(records[:-1])["0.1"]
        self.assertIsNone(incomplete["seed_cluster_bootstrap_95pct"])

if __name__ == "__main__":
    unittest.main()
