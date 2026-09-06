"""Experimental four-player search with explicit opponent policies.

Own turns use PUCT; opponent turns are sampled from a move predictor.
Every simulation starts with a freshly sampled hidden bag and its own game
RNG. Nodes are keyed by observable state, so random future factory layouts
are not frozen into one child. Leaves roll out to actual game completion:
the old network's rank value is deliberately NOT treated as a win chance.
"""
import math
import random
import time

import numpy as np

from agents.sim import AzulSim
from az.actions import legal_mask, move_to_action
from az.encoder import encode
from model import Player


def opponent_distribution(prior, top_k=3, tail_weight=0.15):
    """Concentrate on likely choices while retaining every legal response.

    These are model assumptions, not calibrated human probabilities. Keeping
    the full-prior tail also prevents hard exclusion of a surprising move.
    """
    if not 0 <= tail_weight <= 1 or top_k < 1:
        raise ValueError("top_k must be positive and tail_weight in [0,1]")
    prior = np.asarray(prior, dtype=np.float64)
    if prior.ndim != 1 or not len(prior) or np.any(prior < 0) or not np.isfinite(prior).all() or prior.sum() <= 0:
        raise ValueError("prior must contain finite nonnegative probabilities")
    prior = prior / prior.sum()
    top = np.argsort(-prior, kind="stable")[:top_k]
    result = tail_weight * prior
    result[top] += (1 - tail_weight) * prior[top] / prior[top].sum()
    return result / result.sum()


class MovePredictor:
    def __init__(self, net, top_k=3, tail_weight=0.15):
        self.net = net
        self.top_k = top_k
        self.tail_weight = tail_weight
        opponent_distribution([1.0], top_k, tail_weight)

    def predict(self, gs, pid):
        _, mapping = legal_mask(gs.players[pid].GetAvailableMoves(gs))
        actions = np.asarray(sorted(mapping), dtype=np.int32)
        if not len(actions):
            raise ValueError("No legal moves to predict")
        logits, _ = self.net.forward(encode(gs, pid))
        selected = np.asarray(logits[actions], dtype=np.float64)
        if not np.isfinite(selected).all():
            raise ValueError("Network produced non-finite policy logits")
        # Clipping preserves a nonzero tail even for very confident logits.
        prior = np.exp(np.clip(selected - selected.max(), -60, 0))
        prior /= prior.sum()
        response = opponent_distribution(prior, self.top_k, self.tail_weight)
        return actions, mapping, prior, response


def public_key(sim):
    """Include public tile accounting, but neither bag order nor random state.

    The network combines draw/discard counts and omits floor colours, so its
    encoding alone is insufficient to distinguish future draw distributions.
    """
    gs = sim.gs
    counts = tuple(gs.bag.count(t) for t in range(5))
    used = tuple(gs.bag_used.count(t) for t in range(5))
    floors = tuple(tuple(p.floor_tiles.count(t) for t in range(5)) for p in gs.players)
    return sim.cur, encode(gs, sim.cur).tobytes(), counts, used, floors


class _Node:
    def __init__(self, sim, predictor):
        self.actions, self.mapping, self.prior, self.response = predictor.predict(sim.gs, sim.cur)
        self.visits = np.zeros(len(self.actions), dtype=np.int32)
        self.values = np.zeros(len(self.actions), dtype=np.float64)

    def select(self, rng, own_turn, opponent_mode, c_puct):
        if not own_turn:
            if opponent_mode == "uniform":
                return rng.randrange(len(self.actions))
            if opponent_mode == "greedy":
                return int(np.argmax(self.prior))
            return rng.choices(range(len(self.actions)), weights=self.response, k=1)[0]
        # Neutral first-play value for a four-player win-share objective.
        q = np.full(len(self.actions), 0.25)
        np.divide(self.values, self.visits, out=q, where=self.visits > 0)
        u = c_puct * self.prior * math.sqrt(1 + self.visits.sum()) / (1 + self.visits)
        return int(np.argmax(q + u))


class OpponentSearchPlayer(Player):
    """Model-based best response, not a guarantee against arbitrary opponents.

    The policy net is fixed. Values are sampled final win shares against the
    chosen rollout policies; they are not calibrated human win probabilities.
    """
    def __init__(self, _id, net, iterations=128, seed=None,
                 opponent_mode="learned", top_k=3, tail_weight=0.15,
                 c_puct=1.5, time_budget_s=None, name="opponent-search"):
        super().__init__(_id)
        if iterations < 1 or opponent_mode not in ("learned", "uniform", "greedy"):
            raise ValueError("Invalid simulation count or opponent mode")
        if time_budget_s is not None and time_budget_s <= 0:
            raise ValueError("time_budget_s must be positive")
        self.predictor = MovePredictor(net, top_k, tail_weight)
        self.iterations = iterations
        self.rng = random.Random(seed)
        self.opponent_mode = opponent_mode
        self.c_puct = c_puct
        self.time_budget_s = time_budget_s
        self.name = name
        self.last_stats = None

    def _rollout(self, sim):
        for _ in range(400):
            if sim.terminal:
                return sim.win_values()[self.id]
            actions, mapping, prior, response = self.predictor.predict(sim.gs, sim.cur)
            if sim.cur == self.id or self.opponent_mode == "greedy":
                i = int(np.argmax(prior))
            elif self.opponent_mode == "uniform":
                i = self.rng.randrange(len(actions))
            else:
                i = self.rng.choices(range(len(actions)), weights=response, k=1)[0]
            sim.apply(mapping[int(actions[i])])
        raise RuntimeError("Rollout did not finish; no fabricated terminal outcome recorded")

    def SelectMove(self, moves, game_state):
        if len(game_state.players) != 4:
            raise ValueError("Opponent search currently supports four players only")
        if not moves:
            raise ValueError("No legal moves")
        root_sim = AzulSim(game_state, self.id)
        key = public_key(root_sim)
        root = _Node(root_sim, self.predictor)
        nodes = {key: root}
        start = time.perf_counter()
        done = 0
        for _ in range(self.iterations):
            if done and self.time_budget_s is not None and time.perf_counter() - start >= self.time_budget_s:
                break
            sim = root_sim.sample_hidden(self.rng)
            path = []
            for _depth in range(400):
                if sim.terminal:
                    value = sim.win_values()[self.id]
                    break
                state_key = public_key(sim)
                node = nodes.get(state_key)
                if node is None:
                    nodes[state_key] = _Node(sim, self.predictor)
                    value = self._rollout(sim)
                    break
                i = node.select(self.rng, sim.cur == self.id,
                                self.opponent_mode, self.c_puct)
                path.append((node, i))
                sim.apply(node.mapping[int(node.actions[i])])
            else:
                raise RuntimeError("Search path did not finish")
            for node, i in path:
                node.visits[i] += 1
                node.values[i] += value
            done += 1
        visited = [i for i in range(len(root.actions)) if root.visits[i]]
        ranked = sorted(visited, key=lambda i: (root.visits[i], root.values[i] / root.visits[i]), reverse=True)
        self.last_stats = {
            "iterations": done,
            "root_visits": int(root.visits.sum()),
            "nodes": len(nodes),
            "elapsed_s": time.perf_counter() - start,
            "value_kind": "simulated_win_share",
            "opponent_mode": self.opponent_mode,
            "candidates": [{
                "move": root.mapping[int(root.actions[i])],
                "visits": int(root.visits[i]),
                "avg_value": float(root.values[i] / root.visits[i]),
                "prior": float(root.prior[i]),
            } for i in ranked],
        }
        chosen = int(root.actions[ranked[0]])
        # Return a move from the caller's set, including equivalent floor moves.
        return next(move for move in moves if move_to_action(move) == chosen)
