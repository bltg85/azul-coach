"""Train the Azul net on self-play data and export numpy weights.

Loss = policy cross-entropy (against MCTS visit distribution) + value MSE
(against canonical normalised final scores). Uses GPU if available.

    python -m az.train --data data/sp1.npz data/sp2.npz --epochs 20 --out runs/az.npz
    python -m az.train --data data/*.npz --init runs/az.npz --out runs/az2.npz
"""
import argparse
import os

import numpy as np

from az.net import HIDDEN, build_torch_net, export_to_numpy, NumpyNet


def _load_into_torch(model, numpynet):
    """Warm-start a torch model from a NumpyNet's weights."""
    import torch
    w = numpynet.weights
    sd = model.state_dict()
    li = 0
    for i in range(0, len(HIDDEN) * 2, 2):
        sd[f"trunk.{i}.weight"] = torch.tensor(w[f"w{li}"].T)
        sd[f"trunk.{i}.bias"] = torch.tensor(w[f"b{li}"])
        li += 1
    sd["policy.weight"] = torch.tensor(w["w_pol"].T)
    sd["policy.bias"] = torch.tensor(w["b_pol"])
    sd["value.weight"] = torch.tensor(w["w_val"].T)
    sd["value.bias"] = torch.tensor(w["b_val"])
    model.load_state_dict(sd)


def main():
    import torch
    import torch.nn.functional as F

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--init", default=None, help="warm-start from a NumpyNet .npz")
    ap.add_argument("--entropy", type=float, default=0.0,
                    help="entropy bonus on the policy (guards against collapse "
                         "during self-play; e.g. 0.001)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    Xs, Ps, Vs = [], [], []
    for path in args.data:
        d = np.load(path)
        Xs.append(d["X"]); Ps.append(d["P"]); Vs.append(d["V"])
    X = np.concatenate(Xs); P = np.concatenate(Ps); V = np.concatenate(Vs)
    print(f"Training on {len(X)} samples from {len(args.data)} file(s)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = build_torch_net().to(device)
    if args.init:
        _load_into_torch(model, NumpyNet.load(args.init))
        model.to(device)
        print(f"Warm-started from {args.init}")

    Xt = torch.tensor(X, device=device)
    Pt = torch.tensor(P, device=device)
    Vt = torch.tensor(V, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n = len(X)
    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=device)
        tot_p = tot_v = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            xb, pb, vb = Xt[idx], Pt[idx], Vt[idx]
            logits, value = model(xb)
            logp = F.log_softmax(logits, dim=1)
            policy_loss = -(pb * logp).sum(dim=1).mean()
            value_loss = F.mse_loss(value, vb)
            loss = policy_loss + value_loss
            if args.entropy > 0:
                p = logp.exp()
                entropy = -(p * logp).sum(dim=1).mean()
                loss = loss - args.entropy * entropy
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_p += policy_loss.item() * len(idx)
            tot_v += value_loss.item() * len(idx)
        print(f"  epoch {epoch+1}/{args.epochs}  "
              f"policy_ce={tot_p/n:.4f}  value_mse={tot_v/n:.4f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    export_to_numpy(model).save(args.out)
    print(f"Exported numpy weights to {args.out}")


if __name__ == "__main__":
    main()
