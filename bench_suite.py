"""Run multiple bot-vs-heuristic benches and write results to JSON.

For each config we play N games with seat rotation, recording win rate
and per-game scores. The output feeds plot_bench.py.
"""
import argparse
import gc
import json
import statistics
import time

import _framework_path  # noqa: F401
from model import GameRunner, Player  # noqa: E402
from naive_player import NaivePlayer  # noqa: E402

from agents.heuristic import HeuristicPlayer  # noqa: E402
from agents.mcts import MCTSPlayer  # noqa: E402


class RandomPlayer(Player):
    pass  # base Player.SelectMove already picks at random


def make_player(spec, pid):
    if spec == "random":
        return RandomPlayer(pid)
    if spec == "naive":
        return NaivePlayer(pid)
    if spec == "heuristic":
        return HeuristicPlayer(pid)
    if spec.startswith("mcts:"):
        iters = int(spec.split(":", 1)[1])
        return MCTSPlayer(pid, iterations=iters)
    raise SystemExit(f"unknown: {spec!r}")


def run_one(player_specs, seed):
    players = [make_player(s, i) for i, s in enumerate(player_specs)]
    gr = GameRunner(players, seed)
    activity = gr.Run(False)
    return [activity[i][0] for i in range(len(players))]


def bench(label, subject_spec, baseline_spec, games, base_seed):
    """Run `games` 4-player matches: 1 subject vs 3 baselines, rotating seats."""
    specs = [subject_spec, baseline_spec, baseline_spec, baseline_spec]
    n = len(specs)
    wins_subject = 0
    ties = 0
    subject_scores = []
    all_scores = [[] for _ in specs]
    t0 = time.time()
    for i in range(games):
        rotation = i % n
        rotated = specs[rotation:] + specs[:rotation]
        raw = run_one(rotated, seed=base_seed + i)
        # un-rotate so index 0 always refers to subject
        scores = [0] * n
        for seat, sc in enumerate(raw):
            scores[(seat + rotation) % n] = sc
        for j, s in enumerate(scores):
            all_scores[j].append(s)
        subject_scores.append(scores[0])
        best = max(scores)
        winners = [j for j, s in enumerate(scores) if s == best]
        if len(winners) == 1 and winners[0] == 0:
            wins_subject += 1
        elif len(winners) > 1 and 0 in winners:
            # split tie credit: count as a tie (not a win)
            ties += 1
        gc.collect()
    elapsed = time.time() - t0
    return {
        "label": label,
        "subject_spec": subject_spec,
        "baseline_spec": baseline_spec,
        "games": games,
        "wins": wins_subject,
        "ties": ties,
        "win_rate": wins_subject / games,
        "subject_avg": statistics.mean(subject_scores),
        "subject_median": statistics.median(subject_scores),
        "subject_scores": subject_scores,
        "elapsed_s": elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--out", default="bench_results.json")
    ap.add_argument("--seed", type=int, default=10000)
    args = ap.parse_args()

    # Each config: (label, subject_spec)
    # All benched against 3 baseline heuristic players.
    configs = [
        ("random",   "random"),
        ("naive",    "naive"),
        ("heuristic", "heuristic"),  # baseline-vs-itself = 25% reference
        ("MCTS 100", "mcts:100"),
        ("MCTS 300", "mcts:300"),
        ("MCTS 1000", "mcts:1000"),
        ("MCTS 2000", "mcts:2000"),
    ]

    results = []
    for label, spec in configs:
        print(f"\n=== {label} ({spec}) vs 3x heuristic, {args.games} games ===", flush=True)
        r = bench(label, spec, "heuristic", args.games, args.seed)
        print(f"  wins={r['wins']}/{args.games} ({r['win_rate']*100:.0f}%)  "
              f"avg={r['subject_avg']:.1f}  elapsed={r['elapsed_s']:.0f}s", flush=True)
        results.append(r)
        # Persist incrementally so we don't lose data if we get killed
        with open(args.out, "w") as f:
            json.dump({"games_per_config": args.games, "results": results}, f, indent=2)

    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
