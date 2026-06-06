"""AlphaZero-style PUCT search for 4-player Azul.

Differences from the classic UCT in agents/mcts.py:

* No random rollouts. A leaf is evaluated by the net, which returns a value
  vector (one expected normalised score per seat) and a policy prior.
* Selection uses PUCT: Q + c_puct * P * sqrt(sum N) / (1 + N).
* Per-edge stats (N, W) live on the parent; children are created lazily.

The evaluator is any callable: features (FEATURE_SIZE,) -> (policy_logits
(ACTION_SIZE,), value (NUM_PLAYERS,) in canonical seat order, index 0 = the
player to move). NumpyNet.forward matches this signature.
"""
import numpy as np

from az.actions import ACTION_SIZE, legal_mask
from az.encoder import encode


def _normalise_scores(scores):
    m = max(scores)
    if m <= 0:
        return np.zeros(len(scores), dtype=np.float32)
    return np.asarray([s / m for s in scores], dtype=np.float32)


def placement_values(scores):
    """Rank-based outcome in [0,1]: 1st=1.0 ... last=0.0, ties share the
    average. Unlike score/max this always discriminates between players even
    when everyone scores low — which is what stops the value head collapsing
    to a constant during weak self-play."""
    n = len(scores)
    v = np.zeros(n, dtype=np.float32)
    if n <= 1:
        return v + 1.0
    for i in range(n):
        beat = sum(1 for j in range(n) if scores[i] > scores[j])
        tie = sum(1 for j in range(n) if j != i and scores[i] == scores[j])
        v[i] = (beat + 0.5 * tie) / (n - 1)
    return v


def _masked_softmax(logits, legal_idxs):
    z = np.full(ACTION_SIZE, -np.inf, dtype=np.float32)
    z[legal_idxs] = logits[legal_idxs]
    z -= z[legal_idxs].max()
    e = np.exp(z)
    s = e.sum()
    if s <= 0:
        # degenerate — uniform over legal
        p = np.zeros(ACTION_SIZE, dtype=np.float32)
        p[legal_idxs] = 1.0 / len(legal_idxs)
        return p
    return e / s


class _Node:
    __slots__ = ("sim", "to_move", "terminal", "expanded", "value_vec",
                 "legal", "idx_to_move", "P", "N", "W", "children")

    def __init__(self, sim):
        self.sim = sim
        self.to_move = sim.cur
        self.terminal = sim.terminal
        self.expanded = False
        self.value_vec = None
        self.legal = None
        self.idx_to_move = None
        self.P = None          # np.array ACTION_SIZE priors
        self.N = None          # np.array ACTION_SIZE int
        self.W = None          # np.array (ACTION_SIZE, n_players) value sums
        self.children = None   # dict action_idx -> _Node


def _expand_and_eval(node, evaluator, encode_fn):
    legal = node.sim.legal_moves()
    if not legal:
        node.terminal = True
        node.value_vec = placement_values(node.sim.scores())
        return node.value_vec
    mask, idx_to_move = legal_mask(legal)
    legal_idxs = np.flatnonzero(mask)
    feats = encode_fn(node.sim.gs, node.to_move)
    logits, value_canon = evaluator(feats)
    priors = _masked_softmax(np.asarray(logits, dtype=np.float32), legal_idxs)
    n = len(node.sim.gs.players)
    # canonical value -> absolute-seat value
    v_abs = np.zeros(n, dtype=np.float32)
    for k in range(n):
        v_abs[(node.to_move + k) % n] = value_canon[k]
    node.legal = legal_idxs
    node.idx_to_move = idx_to_move
    node.P = priors
    node.N = np.zeros(ACTION_SIZE, dtype=np.int32)
    node.W = np.zeros((ACTION_SIZE, n), dtype=np.float32)
    node.children = {}
    node.expanded = True
    return v_abs


def _select(node, c_puct):
    legal = node.legal
    sum_n = node.N[legal].sum()
    sqrt_sum = np.sqrt(sum_n) + 1e-8
    q = np.zeros(len(legal), dtype=np.float32)
    nz = node.N[legal] > 0
    q[nz] = node.W[legal][nz, node.to_move] / node.N[legal][nz]
    u = c_puct * node.P[legal] * sqrt_sum / (1 + node.N[legal])
    return legal[int(np.argmax(q + u))]


def _simulate(node, evaluator, c_puct, encode_fn):
    if node.terminal:
        if node.value_vec is None:
            node.value_vec = placement_values(node.sim.scores())
        return node.value_vec
    if not node.expanded:
        return _expand_and_eval(node, evaluator, encode_fn)
    a = _select(node, c_puct)
    child = node.children.get(a)
    if child is None:
        child_sim = node.sim.clone()
        child_sim.apply(node.idx_to_move[a])
        child = _Node(child_sim)
        node.children[a] = child
    v = _simulate(child, evaluator, c_puct, encode_fn)
    node.N[a] += 1
    node.W[a] += v
    return v


def az_search(sim, evaluator, n_sims=200, c_puct=1.5,
              dirichlet_alpha=None, dirichlet_frac=0.25, encode_fn=None):
    """Run PUCT and return (root, visit_counts ACTION_SIZE).

    dirichlet_alpha: if set, add Dirichlet noise to the root priors (used in
    self-play for exploration). Typical alpha ~0.3, frac 0.25.
    encode_fn: state encoder to use (defaults to the current az.encoder.encode);
    pass a different version to evaluate an old net with its own encoder.
    """
    enc = encode_fn or encode
    root = _Node(sim.clone())
    _expand_and_eval(root, evaluator, enc)
    if dirichlet_alpha is not None and len(root.legal) > 0:
        noise = np.random.default_rng().dirichlet([dirichlet_alpha] * len(root.legal))
        root.P[root.legal] = (1 - dirichlet_frac) * root.P[root.legal] + dirichlet_frac * noise
    for _ in range(n_sims):
        _simulate(root, evaluator, c_puct, enc)
    return root, root.N.copy()


def policy_from_visits(visit_counts, temperature=1.0):
    """Normalised move-selection distribution over ACTION_SIZE from visit
    counts. temperature=0 => greedy (argmax); >0 => proportional to N^(1/T)."""
    counts = visit_counts.astype(np.float64)
    if counts.sum() == 0:
        return counts
    if temperature <= 1e-6:
        p = np.zeros_like(counts)
        p[int(np.argmax(counts))] = 1.0
        return p
    c = counts ** (1.0 / temperature)
    return (c / c.sum()).astype(np.float64)
