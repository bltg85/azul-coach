import random
import unittest
from unittest.mock import patch

import webapp
from model import GameState
from agents.sim import AzulSim


class CoachTests(unittest.TestCase):
    def setUp(self):
        self.client = webapp.app.test_client()
        self.client.get("/")
        with self.client.session_transaction() as session:
            self.sid = session["sid"]
        self.game = webapp.GAMES[self.sid]
        gs = GameState(4, rng=random.Random(4))
        for p in gs.players:
            p.player_trace.StartRound()
        self.game["sim"] = AzulSim(gs, 0)
        self.game["bot_specs"] = ["Aragon"] * 3

    def tearDown(self):
        webapp.GAMES.pop(self.sid, None)

    def test_experimental_advice_renders_without_playing_a_move(self):
        sim = self.game["sim"]
        bag = sim.gs.bag[:]
        state = sim.gs.rng.getstate()
        result = self.client.post("/coach", data={"engine": "opponent", "iter": "4"}, follow_redirects=True)
        self.assertEqual(result.status_code, 200)
        self.assertIn(b"simulated win shares", result.data)
        self.assertEqual(self.game["coach"]["iterations"], 4)
        self.assertFalse(self.game["coach_busy"])
        self.assertEqual(sim.gs.bag, bag)
        self.assertEqual(sim.gs.rng.getstate(), state)
        self.assertEqual(self.game["log"], [])

    def test_standard_advice_still_works(self):
        result = self.client.post("/coach", data={"iter": "4"}, follow_redirects=True)
        self.assertEqual(result.status_code, 200)
        self.assertIn(b"relative score estimates", result.data)
        self.assertFalse(self.game["coach"]["experimental"])

    def test_two_player_game_cannot_use_four_player_network(self):
        self.game["sim"] = AzulSim(GameState(2, rng=random.Random(1)), 0)
        self.assertNotIn(b"Try opponent-aware", self.client.get("/").data)
        response = self.client.post("/coach", data={"engine": "opponent"}, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"requires four players", response.data)
        self.assertIsNone(self.game["coach"])

    def test_failure_releases_busy_flag(self):
        with patch.object(webapp.OpponentSearchPlayer, "SelectMove", side_effect=RuntimeError("test")):
            with patch.dict(webapp.app.config, TESTING=True):
                with self.assertRaises(RuntimeError):
                    self.client.post("/coach", data={"engine": "opponent", "iter": "4"})
        self.assertFalse(self.game["coach_busy"])

    def test_stale_analysis_is_not_attached_to_new_game(self):
        original = webapp.OpponentSearchPlayer.SelectMove
        def move_and_replace(bot, moves, gs):
            move = original(bot, moves, gs)
            self.game["sim"] = AzulSim(GameState(4, rng=random.Random(22)), 0)
            return move
        with patch.object(webapp.OpponentSearchPlayer, "SelectMove", move_and_replace):
            self.client.post("/coach", data={"engine": "opponent", "iter": "2"})
        self.assertIsNone(self.game["coach"])

    def test_final_view_uses_row_tiebreak(self):
        sim = self.game["sim"]
        for p in sim.gs.players:
            p.score = 50
        sim.gs.players[2].grid_state[0, :] = 1
        sim.terminal = True
        view = webapp.build_view(self.game)
        self.assertEqual(view["winner_seats"], [2])


if __name__ == "__main__":
    unittest.main()
