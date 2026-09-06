"""Benchmark-only wall-clock adaptation of serving Gumbel sequential halving.

Uses the same top-16 root candidates, scores and non-root search as AZPlayer.
Each halving phase gets an equal fraction of the remaining deadline. The
fixed-simulation serving and training entry points are unchanged.
"""
import math
import random
import time

import numpy as np

from agents.sim import AzulSim
from az.actions import move_to_action
from az.gumbel import _Node, _expand, _root_child_sim, _improved_logits, MAX_CONSIDERED
from model import Player


class TimedAZPlayer(Player):
    def __init__(self, pid, net, seconds, seed=None):
        super().__init__(pid)
        if seconds <= 0:
            raise ValueError("seconds must be positive")
        self.net = net
        self.seconds = seconds
        self.rng = random.Random(seed)
        self.last_stats = None

    def SelectMove(self, moves, game_state):
        start = time.perf_counter()
        if not moves:
            raise ValueError("No legal moves")
        root = _Node(AzulSim(game_state, self.id).sample_hidden(self.rng))
        _expand(root, self.net.forward)
        candidates = list(np.argsort(root.logits)[::-1][:MAX_CONSIDERED])
        phases = max(1, int(math.ceil(math.log2(len(candidates)))))
        deadline = start + self.seconds
        for phase in range(phases):
            if len(candidates) == 1:
                break
            now = time.perf_counter()
            phase_end = now + max(0, deadline - now) / (phases - phase)
            while time.perf_counter() < phase_end:
                # Round-robin allocation. Deadline can stop a partial round;
                # at most one atomic tree simulation overruns the deadline.
                for ci in candidates:
                    if time.perf_counter() >= phase_end:
                        break
                    _root_child_sim(root, ci, self.net.forward)
            scores = _improved_logits(root)
            candidates = sorted(candidates, key=lambda c: scores[c], reverse=True)
            candidates = candidates[:max(1, len(candidates) // 2)]
        chosen = int(root.legal[candidates[0]])
        self.last_stats = {"iterations": int(root.N.sum()),
                           "elapsed_s": time.perf_counter() - start}
        return next(m for m in moves if move_to_action(m) == chosen)
