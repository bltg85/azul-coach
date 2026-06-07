"""Gumbel AlphaZero search (Danihelka et al., ICLR 2022).

Why: standard PUCT can fail to improve the policy when the simulation
budget doesn't visit all root actions — exactly our regime (100+ legal
moves, 150-300 sims). Gumbel guarantees a policy-improvement step even with
very few simulations, which both breaks the plateau and lets us cut sims.

How it differs from agents/mcts.py and az/mcts_az.py:
  * Root: sample m actions WITHOUT replacement via the Gumbel-top-k trick,
    then distribute the simulation budget over them with Sequential Halving,
    scoring by  g + logits + sigma(Q).  The survivor is the move played
    (the Gumbel noise IS the exploration — no separate temperature).
  * Non-root: deterministic selection that tracks the improved policy:
    argmax_a [ pi'(a) - N(a)/(1+sum N) ].
  * Training target: the *completed/improved* policy
    pi'(a) = softmax(logits + sigma(completedQ(a)))  (NO Gumbel noise),
    where unvisited actions get the node's network value as their Q.

Values are per-seat vectors in [0,1] (canonical: index 0 = player to move),
identical to mcts_az, so self-play targets stay compatible.
"""
import math

import numpy as np

from az.actions import ACTION_SIZE, legal_mask
from az.encoder import encode
from az.mcts_az import placement_values

C_VISIT = 50.0
C_SCALE = 1.0
MAX_CONSIDERED = 16  # m: actions sampled at the root


class _Node:
    __slots__ = ("sim", "to_move", "terminal", "expanded", "value_vec",
                 "value_self", "legal", "idx_to_move", "logits",
                 "N", "Wsum", "children")

    def __init__(self, sim):
        self.sim = sim
        self.to_move = sim.cur
        self.terminal = sim.terminal
        self.expanded = False
        self.value_vec = None     # absolute-seat value (for backup)
        self.value_self = 0.0     # net value for to_move (completedQ of unvisited)
        self.legal = None         # array of legal action indices
        self.idx_to_move = None
        self.logits = None        # raw policy logits, per legal action
        self.N = None             # visits, per legal
        self.Wsum = None          # (n_legal, n_players) value sums
        self.children = None      # action_idx -> _Node


def _softmax(x):
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def _expand(node, evaluator):
    """Evaluate a leaf; fill per-legal logits/N/Wsum and the value vector.
    Returns the absolute-seat value vector for backup."""
    legal = node.sim.legal_moves()
    if not legal:
        node.terminal = True
        node.value_vec = placement_values(node.sim.scores())
        return node.value_vec
    mask, idx_to_move = legal_mask(legal)
    legal_idxs = np.flatnonzero(mask)
    raw_logits, value_canon = evaluator(encode(node.sim.gs, node.to_move))
    n = len(node.sim.gs.players)
    v_abs = np.zeros(n, dtype=np.float32)
    for k in range(n):
        v_abs[(node.to_move + k) % n] = value_canon[k]
    node.legal = legal_idxs
    node.idx_to_move = idx_to_move
    node.logits = np.asarray(raw_logits, dtype=np.float64)[legal_idxs]
    node.N = np.zeros(len(legal_idxs), dtype=np.int32)
    node.Wsum = np.zeros((len(legal_idxs), n), dtype=np.float64)
    node.value_self = float(value_canon[0])
    node.value_vec = v_abs
    node.children = {}
    node.expanded = True
    return v_abs


def _improved_logits(node):
    """logits + sigma(completedQ) over this node's legal actions."""
    N = node.N
    visited = N > 0
    Q = np.full(len(N), node.value_self, dtype=np.float64)  # unvisited -> net value
    Q[visited] = node.Wsum[visited, node.to_move] / N[visited]
    sigma = (C_VISIT + (N.max() if len(N) else 0)) * C_SCALE * Q
    return node.logits + sigma


def _simulate(node, evaluator):
    """Descend with deterministic improved-policy selection; expand+backup."""
    if node.terminal:
        if node.value_vec is None:
            node.value_vec = placement_values(node.sim.scores())
        return node.value_vec
    if not node.expanded:
        return _expand(node, evaluator)
    pi = _softmax(_improved_logits(node))
    sumN = node.N.sum()
    ci = int(np.argmax(pi - node.N / (1 + sumN)))
    a = int(node.legal[ci])
    child = node.children.get(a)
    if child is None:
        child_sim = node.sim.clone()
        child_sim.apply(node.idx_to_move[a])
        child = _Node(child_sim)
        node.children[a] = child
    v = _simulate(child, evaluator)
    node.N[ci] += 1
    node.Wsum[ci] += v
    return v


def _root_child_sim(root, ci, evaluator):
    """One root simulation forced through candidate index ci."""
    a = int(root.legal[ci])
    child = root.children.get(a)
    if child is None:
        child_sim = root.sim.clone()
        child_sim.apply(root.idx_to_move[a])
        child = _Node(child_sim)
        root.children[a] = child
    v = _simulate(child, evaluator)
    root.N[ci] += 1
    root.Wsum[ci] += v
    return v


def gumbel_search(sim, evaluator, n_sims=64, rng=None, add_noise=True):
    """Run Gumbel AlphaZero. Returns (best_action_idx, improved_policy, value_vec).

    improved_policy is a length-ACTION_SIZE target (zero on illegal moves).
    add_noise=True (self-play) explores via Gumbel noise; False (eval/serving)
    plays the deterministic strongest move.
    """
    rng = rng or np.random.default_rng()
    root = _Node(sim.clone())
    _expand(root, evaluator)
    n_legal = len(root.legal)
    if n_legal == 0:
        return None, np.zeros(ACTION_SIZE, dtype=np.float32), root.value_vec
    if n_legal == 1:
        target = np.zeros(ACTION_SIZE, dtype=np.float32)
        target[int(root.legal[0])] = 1.0
        return int(root.legal[0]), target, root.value_vec

    m = min(MAX_CONSIDERED, n_legal)
    gumbel = rng.gumbel(size=n_legal) if add_noise else np.zeros(n_legal)
    g_plus_logits = gumbel + root.logits
    # initial candidate set: top-m by g + logits (Gumbel-top-k)
    candidates = list(np.argsort(g_plus_logits)[::-1][:m])

    num_phases = max(1, int(math.ceil(math.log2(m))))
    for _ in range(num_phases):
        if len(candidates) == 1:
            break
        # equal sims per candidate this phase (budget split across phases)
        per = max(1, n_sims // (num_phases * len(candidates)))
        for _s in range(per):
            for ci in candidates:
                _root_child_sim(root, ci, evaluator)
        # score = g + logits + sigma(Q), keep top half
        N = root.N
        visited = N > 0
        Q = np.full(n_legal, root.value_self, dtype=np.float64)
        Q[visited] = root.Wsum[visited, root.to_move] / N[visited]
        sig = (C_VISIT + (N.max() if len(N) else 0)) * C_SCALE * Q
        scores = g_plus_logits + sig
        candidates = sorted(candidates, key=lambda c: scores[c], reverse=True)
        candidates = candidates[: max(1, len(candidates) // 2)]

    best_ci = candidates[0]
    best_action = int(root.legal[best_ci])
    # training target = completed/improved policy (no Gumbel noise)
    improved = _softmax(_improved_logits(root))
    target = np.zeros(ACTION_SIZE, dtype=np.float32)
    target[root.legal] = improved.astype(np.float32)
    return best_action, target, root.value_vec
