"""A framework Player backed by the AlphaZero net + PUCT search.

Used both for evaluation (az vs baselines) and for serving in the web app.
Targets the 4-player game.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (
    os.path.normpath(os.path.join(_HERE, "..", "framework")),
    os.path.normpath(os.path.join(_HERE, "..", "..", "framework")),
):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)
        break

from model import Player  # noqa: E402

from az.mcts_az import az_search  # noqa: E402
from az.net import NumpyNet  # noqa: E402
from agents.sim import AzulSim  # noqa: E402


class AZPlayer(Player):
    def __init__(self, _id, net, n_sims=160, c_puct=1.5, name="azero"):
        super().__init__(_id)
        self.net = net
        self.n_sims = n_sims
        self.c_puct = c_puct
        self.name = name

    @classmethod
    def from_weights(cls, _id, path, **kw):
        return cls(_id, NumpyNet.load(path), **kw)

    def SelectMove(self, moves, game_state):
        sim = AzulSim(game_state, self.id)
        root, visits = az_search(sim, self.net.forward,
                                 n_sims=self.n_sims, c_puct=self.c_puct)
        if visits.sum() == 0:
            return moves[0]
        a = int(np.argmax(visits))
        mv = root.idx_to_move.get(a)
        return mv if mv is not None else moves[0]
