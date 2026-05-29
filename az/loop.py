"""Orchestrate the AlphaZero loop: self-play -> train -> repeat.

Each iteration generates fresh self-play games with the latest net, then
trains on a replay buffer of the most recent iterations' data (warm-started
from the previous net). Runs the verified selfplay/train CLIs as subprocesses.

    python -m az.loop --iters 10 --games 60 --sims 120 --epochs 15 --out-dir runs/r1

Resumable-ish: skips an iteration's step if its output file already exists.
"""
import argparse
import os
import subprocess
import sys
import time


def _run(cmd):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--games", type=int, default=60, help="self-play games per iter")
    ap.add_argument("--sims", type=int, default=120, help="MCTS sims per move")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--buffer", type=int, default=3, help="recent iters to train on")
    ap.add_argument("--workers", type=int, default=1, help="parallel self-play processes")
    ap.add_argument("--out-dir", default="runs/loop")
    ap.add_argument("--init", default=None, help="warm-start weights for iter 0")
    args = ap.parse_args()

    py = sys.executable
    data_dir = os.path.join(args.out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    prev = args.init
    t0 = time.time()

    for i in range(args.iters):
        print(f"\n=== ITERATION {i} ({time.time()-t0:.0f}s elapsed) ===", flush=True)
        data_path = os.path.join(data_dir, f"sp{i}.npz")
        weights_path = os.path.join(args.out_dir, f"az{i}.npz")

        # 1. self-play
        if os.path.exists(data_path):
            print(f"  (self-play data {data_path} exists — skipping)")
        else:
            cmd = [py, "-m", "az.selfplay_az", "--games", str(args.games),
                   "--sims", str(args.sims), "--seed", str(i), "--out", data_path,
                   "--workers", str(args.workers)]
            if prev:
                cmd += ["--weights", prev]
            _run(cmd)

        # 2. train on the replay buffer (this iter + a few previous)
        if os.path.exists(weights_path):
            print(f"  (weights {weights_path} exist — skipping train)")
        else:
            lo = max(0, i - args.buffer + 1)
            recent = [os.path.join(data_dir, f"sp{j}.npz") for j in range(lo, i + 1)]
            recent = [p for p in recent if os.path.exists(p)]
            cmd = [py, "-m", "az.train", "--data", *recent,
                   "--epochs", str(args.epochs), "--out", weights_path]
            if prev:
                cmd += ["--init", prev]
            _run(cmd)

        prev = weights_path

    print(f"\nDone. Latest weights: {prev}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
