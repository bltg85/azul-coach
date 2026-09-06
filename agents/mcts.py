"""Multi-player MCTS for Azul.

Standard UCT with a few specifics for 3-4 player play:

* Each node stores a per-player value sum (one slot per player at the
  table), not a scalar. UCB selection at a node uses the value slot of
  the player about to move at that node.
* Backprop pushes the per-player normalised final scores up the tree.
* Rollouts use a fast policy (default: the level-1 heuristic, or a
  cheap analytical policy for speed). Rollout policy is pluggable.

The default scoring normalises by the winning score so the leader always
gets 1.0, others get fractions. This rewards both winning and high
relative score, which matters for coaching ("would I have scored more
points with this move?").
"""
import math
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (
    os.path.normpath(os.path.join(_HERE, "..", "framework")),
    os.path.normpath(os.path.join(_HERE, "..", "..", "framework")),
):
    if os.path.isdir(_cand):
        if _cand in sys.path:
            sys.path.remove(_cand)
        sys.path.insert(0, _cand)
        break

from model import Player, PlayerState  # noqa: E402
from utils import Move  # noqa: E402

from agents.heuristic import HeuristicPlayer  # noqa: E402
from agents.sim import AzulSim  # noqa: E402


# ---------------------------------------------------------------------------
# Rollout policies. Each takes (player_id, moves, game_state) -> move.
# ---------------------------------------------------------------------------

def random_rollout(_pid, moves, _gs):
    return random.choice(moves)


def fast_heuristic_rollout(pid, moves, gs):
    """Analytical move scoring without deepcopy. Much faster than the full
    HeuristicPlayer. Roughly: floor penalty + pattern-line progress value."""
    plr = gs.players[pid]
    floor_filled = sum(plr.floor)
    best = moves[0]
    best_score = float("-inf")
    for move in moves:
        tg = move[2]
        # Floor penalty
        floor_pen = 0
        for i in range(tg.num_to_floor_line):
            pos = floor_filled + i
            if pos < len(PlayerState.FLOOR_SCORES):
                floor_pen += PlayerState.FLOOR_SCORES[pos]
        # First-player token tile if we're taking from centre first
        if move[0] == Move.TAKE_FROM_CENTRE and not gs.first_player_taken:
            pos = floor_filled + tg.num_to_floor_line
            if pos < len(PlayerState.FLOOR_SCORES):
                floor_pen += PlayerState.FLOOR_SCORES[pos]
        # Pattern-line value
        line_val = 0.0
        if tg.num_to_pattern_line > 0:
            line = tg.pattern_line_dest
            line_size = line + 1
            new_total = plr.lines_number[line] + tg.num_to_pattern_line
            if new_total == line_size:
                # Completes — rough placement value
                line_val = 1 + line * 0.5
            else:
                line_val = (new_total / line_size) * (1 + line * 0.5)
        score = line_val + floor_pen
        if score > best_score:
            best_score = score
            best = move
    return best


# ---------------------------------------------------------------------------
# Node + search.
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = ("sim", "parent", "move", "children", "untried", "visits", "value_sum")

    def __init__(self, sim, parent, move, rng):
        self.sim = sim
        self.parent = parent
        self.move = move
        self.children = []
        self.untried = sim.legal_moves() if not sim.terminal else []
        rng.shuffle(self.untried)  # avoid deterministic expansion order bias
        self.visits = 0
        self.value_sum = [0.0] * len(sim.gs.players)


def _normalise_scores(scores):
    m = max(scores)
    if m <= 0:
        return [0.0] * len(scores)
    return [s / m for s in scores]


def _ucb_select(node, c):
    pid = node.sim.cur
    log_n = math.log(node.visits)
    best = None
    best_score = float("-inf")
    for ch in node.children:
        avg = ch.value_sum[pid] / ch.visits
        ucb = avg + c * math.sqrt(log_n / ch.visits)
        if ucb > best_score:
            best_score = ucb
            best = ch
    return best


ROLLOUT_PLY_LIMIT = 300  # safety net; normal games end in <100 plies


def _rollout(sim, policy, rng):
    sim = sim.clone()
    plies = 0
    while not sim.terminal and plies < ROLLOUT_PLY_LIMIT:
        moves = sim.legal_moves()
        if not moves:
            break  # defensive — shouldn't happen but don't loop forever
        move = (rng.choice(moves) if policy is random_rollout
                else policy(sim.cur, moves, sim.gs))
        sim.apply(move)
        plies += 1
    # If we bailed out before terminal, give end-of-game bonuses on the
    # current state so the value signal still reflects wall progress.
    if not sim.terminal:
        for p in sim.gs.players:
            p.EndOfGameScore()
    return _normalise_scores(sim.scores())


def _backprop(node, normalised_scores):
    while node is not None:
        node.visits += 1
        for i, v in enumerate(normalised_scores):
            node.value_sum[i] += v
        node = node.parent


def mcts_search(root_sim, iterations=None, time_budget_s=None,
                rollout_policy=fast_heuristic_rollout, c=1.4,
                root_moves=None, rng=None):
    """Run MCTS. Provide either iterations OR time_budget_s.

    Returns (root_node, num_iterations_done).

    root_moves: if given, overrides the legal-move set at the root. Useful
    for top-K pruning where the heuristic preselects candidate moves.
    """
    if iterations is None and time_budget_s is None:
        iterations = 500
    rng = rng if rng is not None else random.Random()
    root = _Node(root_sim.sample_hidden(rng), parent=None, move=None, rng=rng)
    if root_moves is not None:
        root.untried = list(root_moves)
        rng.shuffle(root.untried)
    deadline = time.time() + time_budget_s if time_budget_s is not None else None
    iters_done = 0
    while True:
        if iterations is not None and iters_done >= iterations:
            break
        if deadline is not None and time.time() >= deadline:
            break

        node = root
        # Selection
        while not node.sim.terminal and not node.untried and node.children:
            node = _ucb_select(node, c)
        # Expansion
        if not node.sim.terminal and node.untried:
            move = node.untried.pop()
            child_sim = node.sim.clone()
            child_sim.apply(move)
            child = _Node(child_sim, parent=node, move=move, rng=rng)
            node.children.append(child)
            node = child
        # Rollout
        if node.sim.terminal:
            scores = _normalise_scores(node.sim.scores())
        else:
            scores = _rollout(node.sim, rollout_policy, rng)
        _backprop(node, scores)
        iters_done += 1
    return root, iters_done


# ---------------------------------------------------------------------------
# Player.
# ---------------------------------------------------------------------------

class MCTSPlayer(Player):
    def __init__(self, _id, iterations=1000, time_budget_s=None,
                 rollout_policy=fast_heuristic_rollout, c=1.4, name="mcts",
                 top_k=20, seed=None):
        super().__init__(_id)
        self.iterations = iterations
        self.time_budget_s = time_budget_s
        self.rollout_policy = rollout_policy
        self.c = c
        self.name = name
        self.top_k = top_k
        self.rng = random.Random(seed)
        self._heuristic = HeuristicPlayer(_id)
        self.last_stats = None  # populated after each SelectMove for coaching

    def _prune(self, moves, game_state):
        if self.top_k is None or len(moves) <= self.top_k:
            return moves
        self._heuristic.id = self.id
        scored = [(self._heuristic._evaluate(m, game_state), m) for m in moves]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[: self.top_k]]

    def SelectMove(self, moves, game_state):
        # The framework hands us deepcopies, so we own them.
        sim = AzulSim(game_state, self.id)
        pruned = self._prune(moves, game_state)
        root, iters = mcts_search(
            sim,
            iterations=self.iterations,
            time_budget_s=self.time_budget_s,
            rollout_policy=self.rollout_policy,
            c=self.c,
            root_moves=pruned,
            rng=self.rng,
        )
        if not root.children:
            return moves[0]  # only happens if no legal moves (shouldn't)

        # Robust child: most visited (more stable than highest-value)
        best = max(root.children, key=lambda ch: ch.visits)
        # Store stats for coach consumption
        ranked = sorted(
            root.children,
            key=lambda ch: ch.visits,
            reverse=True,
        )
        self.last_stats = {
            "iterations": iters,
            "root_visits": root.visits,
            "candidates": [
                {
                    "move": ch.move,
                    "visits": ch.visits,
                    "avg_value": ch.value_sum[self.id] / ch.visits,
                }
                for ch in ranked
            ],
        }
        return best.move
