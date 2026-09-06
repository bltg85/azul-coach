"""Reproducible pilot: model-based search vs three legacy network bots.

Rotate the subject through all four seats for each deal seed. Search RNGs
are separate from the game's RNG. No games are silently truncated. This is
a bot-only benchmark, not validation against humans or tournament players.
"""
import os
# Small single-position matrix products should not launch large BLAS pools.
# Set before importing numpy; subprocess workers inherit this configuration.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import _framework_path
from model import GameState
from agents.sim import AzulSim
from agents.opponent_search import OpponentSearchPlayer, MovePredictor
from az.actions import move_to_action
from az.net import NumpyNet
from az.player import AZPlayer
import numpy as np


def play(task):
    config, game, opts = task
    deal_seed = opts["seed"] + game // 4
    subject = game % 4
    net = NumpyNet.load(opts["weights"])
    predictor = MovePredictor(net)
    bots = [AZPlayer(i, net, n_sims=opts["baseline_sims"], algo="gumbel",
                     seed=deal_seed * 1009 + i) for i in range(4)]
    if config != "az":
        bots[subject] = OpponentSearchPlayer(
            subject, net, iterations=opts["sims"],
            seed=deal_seed * 1009 + subject, opponent_mode=config)
    gs = GameState(4, rng=random.Random(deal_seed))
    for p in gs.players:
        p.player_trace.StartRound()
    sim = AzulSim(gs, gs.first_player)
    timings = []
    coverage = {"positions": 0, "top1": 0, "top3": 0, "top3_mass_sum": 0.0}
    start = time.perf_counter()
    for ply in range(400):
        if sim.terminal:
            break
        moves = sim.legal_moves()
        pid = sim.cur
        # Only the all-AZ control games measure prediction of search choices.
        prediction = predictor.predict(sim.gs, pid) if config == "az" else None
        t0 = time.perf_counter()
        move = bots[pid].SelectMove(moves, sim.gs)
        if pid == subject:
            timings.append(time.perf_counter() - t0)
        if prediction is not None:
            actions, _, prior, _ = prediction
            order = np.argsort(-prior, kind="stable")
            action = move_to_action(move)
            coverage["positions"] += 1
            coverage["top1"] += int(action == actions[order[0]])
            coverage["top3"] += int(action in actions[order[:3]])
            coverage["top3_mass_sum"] += float(prior[order[:3]].sum())
        sim.apply(move)
    if not sim.terminal:
        raise RuntimeError(f"Unfinished game {config}/{game}; no outcome recorded")
    return {
        "config": config, "game": game, "deal_seed": deal_seed,
        "subject_seat": subject, "scores": sim.scores(),
        "completed_rows": [p.GetCompletedRows() for p in sim.gs.players],
        "winners": sim.winners(), "win_share": sim.win_values()[subject],
        "subject_move_seconds": timings, "prediction": coverage,
        "elapsed_s": time.perf_counter() - start,
    }


def summarize(records):
    result = {}
    for config in sorted({r["config"] for r in records}):
        rows = [r for r in records if r["config"] == config]
        times = [t for r in rows for t in r["subject_move_seconds"]]
        # The all-AZ control repeats the same game while rotating which seat
        # is scored. Count its predictions only once per deal seed.
        prediction_rows = [r for r in rows if r["subject_seat"] == 0]
        count = sum(r["prediction"]["positions"] for r in prediction_rows)
        result[config] = {
            "games": len(rows),
            "outright_wins": sum(r["winners"] == [r["subject_seat"]] for r in rows),
            "ties_for_first": sum(r["subject_seat"] in r["winners"] and len(r["winners"]) > 1 for r in rows),
            "mean_win_share": statistics.mean(r["win_share"] for r in rows),
            "mean_score": statistics.mean(r["scores"][r["subject_seat"]] for r in rows),
            "mean_move_s": statistics.mean(times),
            "p95_move_s": float(np.quantile(times, .95)),
        }
        if count:
            result[config]["prediction"] = {
                "positions": count,
                "top1_hit_rate": sum(r["prediction"]["top1"] for r in prediction_rows) / count,
                "top3_hit_rate": sum(r["prediction"]["top3"] for r in prediction_rows) / count,
                "mean_prior_top3_mass": sum(r["prediction"]["top3_mass_sum"] for r in prediction_rows) / count,
                "target": "legacy Gumbel search using the same network; NOT humans",
            }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--baseline-sims", type=int, default=80)
    ap.add_argument("--seed", type=int, default=70000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--weights", default="az/weights/az_v1.npz")
    ap.add_argument("--configs", nargs="+", default=["az", "learned", "greedy", "uniform"],
                    choices=["az", "learned", "greedy", "uniform"])
    ap.add_argument("--out", default="runs/opponent-pilot.json")
    args = ap.parse_args()
    if args.games < 4 or args.games % 4 or min(args.sims, args.baseline_sims, args.workers) < 1:
        ap.error("Use a positive multiple of four games and positive budgets/workers")
    opts = vars(args)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = [(c, g, opts) for c in args.configs for g in range(args.games)]
        futures = [pool.submit(play, task) for task in tasks]
        for future in as_completed(futures):
            records.append(future.result())
            report = {
                "settings": opts,
                "weights_sha256": hashlib.sha256(Path(args.weights).read_bytes()).hexdigest(),
                "note": "Exploratory bot-only results. AZ and rollout search have different compute costs. Learned/greedy/uniform use equal simulation counts, not equal wall time. Seats share deal seeds; games are not independent samples.",
                "summary": summarize(records),
                "records": sorted(records, key=lambda r: (r["config"], r["game"])),
            }
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            if len(records) % 4 == 0:
                print(f"{len(records)}/{len(tasks)} games, {time.perf_counter()-start:.0f}s", flush=True)
    print(json.dumps(summarize(records), indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
