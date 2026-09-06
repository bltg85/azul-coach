"""Equal per-move wall-clock budgets; resume checkpoints after interruption."""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import _framework_path
from model import GameState
from agents.sim import AzulSim
from agents.opponent_search import OpponentSearchPlayer
from az.timed_player import TimedAZPlayer
from az.net import NumpyNet
from az.actions import legal_mask


def play(task):
    game, budget, opts = task
    seed, seat = opts["seed"] + game // 4, game % 4
    net = NumpyNet.load(opts["weights"])
    # Warm inference before either algorithm is timed.
    net.forward(np.zeros(309, dtype=np.float32))
    bots = [TimedAZPlayer(i, net, budget, seed * 1009 + i) for i in range(4)]
    bots[seat] = OpponentSearchPlayer(seat, net, iterations=1000000,
                                    seed=seed * 1009 + seat, time_budget_s=budget)
    gs = GameState(4, rng=random.Random(seed))
    for p in gs.players:
        p.player_trace.StartRound()
    sim = AzulSim(gs, gs.first_player)
    measurements = []
    for ply in range(400):
        if sim.terminal:
            break
        pid = sim.cur
        moves = sim.legal_moves()
        forced = len(legal_mask(moves)[1]) == 1
        start = time.perf_counter()
        move = moves[0] if forced else bots[pid].SelectMove(moves, sim.gs)
        elapsed = time.perf_counter() - start
        measurements.append({"seat": pid, "seconds": elapsed, "forced": forced,
                             "simulations": 0 if forced else bots[pid].last_stats["iterations"]})
        sim.apply(move)
    if not sim.terminal:
        raise RuntimeError("Unfinished game; refusing to record a result")
    return {"game": game, "budget_s": budget, "deal_seed": seed,
            "subject_seat": seat, "scores": sim.scores(),
            "completed_rows": [p.GetCompletedRows() for p in sim.gs.players],
            "winners": sim.winners(), "win_share": sim.win_values()[seat],
            "moves": measurements}


def summarize(records):
    summaries = {}
    for budget in sorted({r["budget_s"] for r in records}):
        rows = [r for r in records if r["budget_s"] == budget]
        seeds = sorted({r["deal_seed"] for r in rows})
        # Bootstrap whole deal-seed blocks, keeping correlated seats together.
        blocks = [np.mean([r["win_share"] for r in rows if r["deal_seed"] == seed]) for seed in seeds]
        complete = all(sum(r["deal_seed"] == seed for r in rows) == 4 for seed in seeds)
        ci = None
        if complete and len(blocks) >= 2:
            rng = np.random.default_rng(90317)
            samples = rng.choice(blocks, size=(10000, len(blocks)), replace=True).mean(axis=1)
            ci = np.quantile(samples, [.025, .975]).tolist()
        timings = {}
        for label in ("learned", "az"):
            moves = [m for r in rows for m in r["moves"]
                     if not m.get("forced", False) and (m["seat"] == r["subject_seat"]) == (label == "learned")]
            seconds = [m["seconds"] for m in moves]
            timings[label] = {"moves": len(moves), "mean_s": float(np.mean(seconds)),
                              "p95_s": float(np.quantile(seconds, .95)),
                              "max_s": max(seconds),
                              "mean_simulations": float(np.mean([m["simulations"] for m in moves])),
                              "fraction_over_budget_by_10pct": float(np.mean(np.array(seconds) > budget * 1.1))}
        summaries[str(budget)] = {"games": len(rows), "deal_seeds": len(seeds),
            "outright_wins": sum(r["winners"] == [r["subject_seat"]] for r in rows),
            "shared_firsts": sum(r["subject_seat"] in r["winners"] and len(r["winners"]) > 1 for r in rows),
            "win_share": float(np.mean([r["win_share"] for r in rows])),
            "seed_cluster_bootstrap_95pct": ci,
            "seat_win_shares": {str(seat): float(np.mean([r["win_share"] for r in rows if r["subject_seat"] == seat]))
                                for seat in sorted({r["subject_seat"] for r in rows})},
            "timings": timings}
    return summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=48, help="Per budget; multiple of four")
    ap.add_argument("--budgets", type=float, nargs="+", default=[.2, .5])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=91000)
    ap.add_argument("--weights", default="az/weights/az_v1.npz")
    ap.add_argument("--out", default="runs/equal-time.json")
    args = ap.parse_args()
    if args.games < 4 or args.games % 4 or args.workers < 1 or any(b <= 0 for b in args.budgets):
        ap.error("Positive budgets/workers and games divisible by four required")
    opts = vars(args)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    weight_hash = hashlib.sha256(Path(args.weights).read_bytes()).hexdigest()
    code_hash = hashlib.sha256(b"".join(Path(p).read_bytes() for p in
        ("bench_equal_time.py", "az/timed_player.py", "agents/opponent_search.py", "az/gumbel.py"))).hexdigest()
    records = []
    if path.exists():
        old = json.loads(path.read_text())
        if old["settings"] != opts or old["weights_sha256"] != weight_hash or old["code_sha256"] != code_hash:
            raise ValueError("Existing checkpoint has different settings/code/weights; use another output path")
        records = old["records"]
    done = {(r["game"], r["budget_s"]) for r in records}
    # Interleave budgets so long-running conditions are not all queued last.
    tasks = [(g, b, opts) for g in range(args.games) for b in args.budgets if (g, b) not in done]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(play, task) for task in tasks]):
            records.append(future.result())
            report = {"settings": opts, "weights_sha256": weight_hash, "code_sha256": code_hash,
                      "python": platform.python_version(), "platform": platform.platform(),
                      "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                      "note": "One learned-policy rollout search vs three time-adapted AZ Gumbel bots. Identical per-move deadline including setup; atomic simulations may overrun. Forced single-action moves bypass both searches and are excluded from timing summaries. Wall-time execution is not bit-reproducible. Seats share seeds; bootstrap clusters by deal seed. Bot-only evidence, not human strength.",
                      "summary": summarize(records),
                      "records": sorted(records, key=lambda r: (r["budget_s"], r["game"]))}
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(report, indent=2), encoding="utf-8")
            temp.replace(path)
            if len(records) % 4 == 0:
                print(f"{len(records)}/{args.games * len(args.budgets)} games, {time.perf_counter()-started:.0f}s", flush=True)
    print(json.dumps(summarize(records), indent=2))

if __name__ == "__main__":
    main()
