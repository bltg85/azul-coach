"""Bench: pit agents against each other across N games and report win-rates.

Usage:
    python selfplay.py                       # default: heuristic vs 3x naive, 200 games
    python selfplay.py --games 500 --players heuristic,naive,naive
"""
import argparse
import gc
import os
import statistics
import sys
import time

import _framework_path  # noqa: F401  (side effect: adds framework/ to sys.path)

from model import GameRunner, Player  # noqa: E402
from naive_player import NaivePlayer  # noqa: E402

from agents.heuristic import HeuristicPlayer  # noqa: E402
from agents.mcts import MCTSPlayer  # noqa: E402


class RandomPlayer(Player):
    """Base Player already picks uniformly at random — but give it a name."""

    def __init__(self, _id):
        super().__init__(_id)
        self.name = "random"


def make_player(spec, pid):
    spec = spec.strip().lower()
    if spec == "heuristic":
        return HeuristicPlayer(pid)
    if spec == "naive":
        p = NaivePlayer(pid)
        p.name = "naive"
        return p
    if spec == "random":
        return RandomPlayer(pid)
    if spec.startswith("mcts"):
        # accept "mcts" or "mcts:300" (iterations) or "mcts:t2" (time seconds)
        iters = 300
        tbudget = None
        if ":" in spec:
            arg = spec.split(":", 1)[1]
            if arg.startswith("t"):
                tbudget = float(arg[1:])
                iters = None
            else:
                iters = int(arg)
        return MCTSPlayer(pid, iterations=iters, time_budget_s=tbudget)
    raise SystemExit(f"unknown player type: {spec!r}")


def run_match(player_specs, seed):
    players = [make_player(spec, i) for i, spec in enumerate(player_specs)]
    gr = GameRunner(players, seed)
    # Suppress framework's verbose Run output by redirecting stdout
    activity = gr.Run(False)
    scores = [activity[i][0] for i in range(len(players))]
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--players",
        default="heuristic,naive,naive,naive",
        help="comma-separated player types (heuristic|naive|random), 2-4 players",
    )
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0, help="base seed; each game uses base+i")
    args = ap.parse_args()

    specs = [s.strip() for s in args.players.split(",")]
    if not (2 <= len(specs) <= 4):
        raise SystemExit("need 2-4 players")

    wins = [0] * len(specs)
    ties = 0
    score_lists = [[] for _ in specs]

    t0 = time.time()
    for i in range(args.games):
        # Rotate seat order so seat advantage averages out
        rotation = i % len(specs)
        rotated = specs[rotation:] + specs[:rotation]
        scores = run_match(rotated, seed=args.seed + i)

        # Un-rotate so wins[i] always refers to specs[i]
        unrotated = [0] * len(specs)
        for seat, sc in enumerate(scores):
            original_idx = (seat + rotation) % len(specs)
            unrotated[original_idx] = sc
        scores = unrotated

        best = max(scores)
        winners = [j for j, s in enumerate(scores) if s == best]
        if len(winners) == 1:
            wins[winners[0]] += 1
        else:
            ties += 1
        for j, s in enumerate(scores):
            score_lists[j].append(s)

        # Free per-game state aggressively; MCTS trees are large.
        gc.collect()

        if (i + 1) % 5 == 0 or args.games <= 10:
            elapsed = time.time() - t0
            print(f"  ... {i+1}/{args.games} games  ({elapsed:.1f}s)", flush=True)

    elapsed = time.time() - t0
    print()
    print(f"Played {args.games} games in {elapsed:.1f}s ({elapsed/args.games*1000:.0f} ms/game)")
    print(f"Setup: {specs}")
    print()
    print(f"{'player':<12} {'wins':>6} {'win%':>7} {'avg':>7} {'median':>7} {'best':>5}")
    for j, spec in enumerate(specs):
        w = wins[j]
        scores = score_lists[j]
        print(
            f"{spec:<12} {w:>6} {w/args.games*100:>6.1f}% "
            f"{statistics.mean(scores):>7.1f} {statistics.median(scores):>7.1f} {max(scores):>5}"
        )
    print(f"{'ties':<12} {ties:>6} {ties/args.games*100:>6.1f}%")


if __name__ == "__main__":
    main()
