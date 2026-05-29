"""The Azul value+policy network.

Two implementations of the same architecture:

* ``AzulNet`` — a PyTorch MLP, used for *training* (GPU). Imported lazily so
  this module loads even when torch isn't installed.
* ``NumpyNet`` — a dependency-free forward pass that loads weights exported
  from a trained ``AzulNet``. Used for *serving* (the web app only has
  numpy) and for fast self-play inference.

Architecture: FEATURE_SIZE -> [256, 256] ReLU -> two heads
    policy head: ACTION_SIZE logits (masked + softmaxed at use)
    value head : NUM_PLAYERS sigmoid outputs in [0,1], in canonical seat
                 order (index 0 = player to move). Target = each seat's
                 final score / winner's score.
"""
import numpy as np

from az import NUM_PLAYERS
from az.actions import ACTION_SIZE
from az.encoder import FEATURE_SIZE

HIDDEN = (256, 256)


# ---------------------------------------------------------------------------
# Numpy inference (no torch).
# ---------------------------------------------------------------------------

def _relu(x):
    return np.maximum(x, 0.0)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class NumpyNet:
    """Forward-only net from exported weights. Accepts a single feature
    vector (FEATURE_SIZE,) or a batch (B, FEATURE_SIZE)."""

    def __init__(self, weights):
        # weights: dict with keys w0,b0,w1,b1,...,w_pol,b_pol,w_val,b_val
        self.weights = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}
        self.n_hidden = sum(1 for k in self.weights if k.startswith("w") and k[1:].isdigit())

    def forward(self, x):
        single = x.ndim == 1
        if single:
            x = x[None, :]
        h = x
        for i in range(self.n_hidden):
            h = _relu(h @ self.weights[f"w{i}"] + self.weights[f"b{i}"])
        policy = h @ self.weights["w_pol"] + self.weights["b_pol"]
        value = _sigmoid(h @ self.weights["w_val"] + self.weights["b_val"])
        if single:
            return policy[0], value[0]
        return policy, value

    def save(self, path):
        np.savez(path, **self.weights)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        return cls({k: data[k] for k in data.files})


def random_numpynet(seed=0):
    """A randomly-initialised NumpyNet — used to bootstrap the very first
    self-play iteration (and to test the search without a trained model)."""
    rng = np.random.default_rng(seed)
    dims = [FEATURE_SIZE, *HIDDEN]
    w = {}
    for i in range(len(HIDDEN)):
        fan_in = dims[i]
        scale = np.sqrt(2.0 / fan_in)
        w[f"w{i}"] = (rng.standard_normal((dims[i], dims[i + 1])) * scale).astype(np.float32)
        w[f"b{i}"] = np.zeros(dims[i + 1], dtype=np.float32)
    last = HIDDEN[-1]
    w["w_pol"] = (rng.standard_normal((last, ACTION_SIZE)) * np.sqrt(2.0 / last)).astype(np.float32)
    w["b_pol"] = np.zeros(ACTION_SIZE, dtype=np.float32)
    w["w_val"] = (rng.standard_normal((last, NUM_PLAYERS)) * np.sqrt(2.0 / last)).astype(np.float32)
    w["b_val"] = np.zeros(NUM_PLAYERS, dtype=np.float32)
    return NumpyNet(w)


# ---------------------------------------------------------------------------
# Torch model (training). Lazily imported.
# ---------------------------------------------------------------------------

def build_torch_net():
    """Construct the PyTorch AzulNet. Raises ImportError if torch missing."""
    import torch
    import torch.nn as nn

    class AzulNet(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            prev = FEATURE_SIZE
            for h in HIDDEN:
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            self.trunk = nn.Sequential(*layers)
            self.policy = nn.Linear(prev, ACTION_SIZE)
            self.value = nn.Linear(prev, NUM_PLAYERS)

        def forward(self, x):
            h = self.trunk(x)
            return self.policy(h), torch.sigmoid(self.value(h))

    return AzulNet()


def export_to_numpy(torch_model):
    """Convert a trained AzulNet's weights to a NumpyNet."""
    sd = torch_model.state_dict()
    w = {}
    # trunk Linear layers live at trunk.0, trunk.2, ... (ReLU has no params)
    li = 0
    for i in range(0, len(HIDDEN) * 2, 2):
        w[f"w{li}"] = sd[f"trunk.{i}.weight"].cpu().numpy().T
        w[f"b{li}"] = sd[f"trunk.{i}.bias"].cpu().numpy()
        li += 1
    w["w_pol"] = sd["policy.weight"].cpu().numpy().T
    w["b_pol"] = sd["policy.bias"].cpu().numpy()
    w["w_val"] = sd["value.weight"].cpu().numpy().T
    w["b_val"] = sd["value.bias"].cpu().numpy()
    return NumpyNet(w)
