"""Measure AZ-net strength vs a baseline in 4-player games.

One AZ seat against three baseline opponents, rotating the AZ seat each game
so seat advantage averages out. Reports AZ win-rate and average score.

    python -m az.evaluate --weights runs/loop/az11.npz --sims 160 \
        --baseline heuristic --games 40
    python -m az.evaluate --weights runs/loop/az11.npz --baseline mcts:200 --games 20
"""
import argparse
import statistics
import time

import _framework_path  # noqa: F401
from model import GameRunner  # noqa: E402

from agents.heuristic import HeuristicPlayer  # noqa: E402
from agents.mcts import MCTSPlayer  # noqa: E402
from az.net import NumpyNet  # noqa: E402
from az.player import AZPlayer  # noqa: E402


_AZ_BASELINE_CACHE = {}


def make_baseline(spec, pid):
    spec = spec.strip()
    low = spec.lower()
    if low == "heuristic":
        return HeuristicPlayer(pid)
    if low.startswith("mcts:"):
        return MCTSPlayer(pid, iterations=int(spec.split(":", 1)[1]))
    if low.startswith("az:") or low.startswith("azv1:"):
        # az:<path>[:sims]   -> net with the CURRENT encoder
        # azv1:<path>[:sims] -> net with the FROZEN v1 encoder (e.g. az20)
        v1 = low.startswith("azv1:")
        parts = spec.split(":")
        path = parts[1]
        sims = int(parts[2]) if len(parts) > 2 else 80
        if path not in _AZ_BASELINE_CACHE:
            _AZ_BASELINE_CACHE[path] = NumpyNet.load(path)
        enc = None
        if v1:
            from az.encoder_v1 import encode as enc
        return AZPlayer(pid, _AZ_BASELINE_CACHE[path], n_sims=sims, encode_fn=enc)
    raise SystemExit(f"unknown baseline {spec!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--sims", type=int, default=160)
    ap.add_argument("--baseline", default="heuristic")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--seed", type=int, default=10000)
    args = ap.parse_args()

    net = NumpyNet.load(args.weights)
    wins = ties = 0
    az_scores = []
    t0 = time.time()
    for g in range(args.games):
        az_seat = g % 4
        players = []
        for i in range(4):
            if i == az_seat:
                players.append(AZPlayer(i, net, n_sims=args.sims))
            else:
                players.append(make_baseline(args.baseline, i))
        activity = GameRunner(players, args.seed + g).Run(False)
        scores = [activity[i][0] for i in range(4)]
        az = scores[az_seat]
        az_scores.append(az)
        best = max(scores)
        winners = [i for i, s in enumerate(scores) if s == best]
        if winners == [az_seat]:
            wins += 1
        elif az_seat in winners:
            ties += 1
        if (g + 1) % 5 == 0 or args.games <= 10:
            print(f"  {g+1}/{args.games}  az_wins={wins} ties={ties} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print()
    print(f"AZ ({args.weights}, {args.sims} sims) vs 3x {args.baseline}, "
          f"{args.games} games:")
    print(f"  win-rate (outright): {wins/args.games*100:.1f}%  "
          f"(+{ties} ties for first)")
    print(f"  AZ avg score: {statistics.mean(az_scores):.1f}  "
          f"median: {statistics.median(az_scores):.1f}  max: {max(az_scores)}")
    # Note: random chance for one of four players is 25%.


if __name__ == "__main__":
    main()
