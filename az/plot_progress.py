"""Plot AlphaZero training progress.

Two views, either or both:

* loss curve   — parsed from the loop's stdout log (policy CE + value MSE,
                 final epoch of each iteration vs iteration number)
* strength     — win-rate + avg score of each iteration's net vs a baseline,
                 measured by playing games (slow; run after training)

    # loss only, from a saved log file
    python -m az.plot_progress --log run.log --out runs/r1/progress.png

    # strength only, evaluating every az*.npz in a run dir
    python -m az.plot_progress --strength --run-dir runs/r1 \
        --baseline heuristic --eval-games 20 --eval-sims 80 \
        --out runs/r1/strength.png

    # both
    python -m az.plot_progress --log run.log --strength --run-dir runs/r1 \
        --out runs/r1/progress.png
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_losses(log_path):
    """Return (iters, policy_ce, value_mse) using the last epoch per iter."""
    iter_re = re.compile(r"=== ITERATION (\d+)")
    loss_re = re.compile(r"policy_ce=([\d.]+)\s+value_mse=([\d.]+)")
    cur = None
    last = {}
    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            mi = iter_re.search(line)
            if mi:
                cur = int(mi.group(1))
                continue
            ml = loss_re.search(line)
            if ml and cur is not None:
                last[cur] = (float(ml.group(1)), float(ml.group(2)))
    iters = sorted(last)
    return iters, [last[i][0] for i in iters], [last[i][1] for i in iters]


def eval_strength(run_dir, baseline, games, sims, seed=20000):
    import _framework_path  # noqa: F401
    from model import GameRunner
    from az.net import NumpyNet
    from az.player import AZPlayer
    from az.evaluate import make_baseline

    paths = [p for p in glob.glob(os.path.join(run_dir, "az*.npz"))
             if re.search(r"az(\d+)\.npz", os.path.basename(p))]
    paths.sort(key=lambda p: int(re.search(r"az(\d+)\.npz", os.path.basename(p)).group(1)))
    iters, winrates, avgscores = [], [], []
    for p in paths:
        it = int(re.search(r"az(\d+)\.npz", p).group(1))
        net = NumpyNet.load(p)
        wins, scores = 0, []
        for g in range(games):
            seat = g % 4
            players = [AZPlayer(i, net, n_sims=sims) if i == seat
                       else make_baseline(baseline, i) for i in range(4)]
            activity = GameRunner(players, seed + g).Run(False)
            sc = [activity[i][0] for i in range(4)]
            scores.append(sc[seat])
            if sc[seat] == max(sc) and sc.count(max(sc)) == 1:
                wins += 1
        iters.append(it)
        winrates.append(wins / games * 100)
        avgscores.append(sum(scores) / len(scores))
        print(f"  az{it}: win-rate {winrates[-1]:.0f}%  avg {avgscores[-1]:.1f}", flush=True)
    return iters, winrates, avgscores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="loop stdout log for loss curve")
    ap.add_argument("--strength", action="store_true")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--baseline", default="heuristic")
    ap.add_argument("--eval-games", type=int, default=20)
    ap.add_argument("--eval-sims", type=int, default=80)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    panels = []
    if args.log:
        panels.append("loss")
    if args.strength:
        panels.append("strength")
    if not panels:
        raise SystemExit("nothing to plot: pass --log and/or --strength")

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.5))
    if len(panels) == 1:
        axes = [axes]

    for ax, panel in zip(axes, panels):
        if panel == "loss":
            it, pce, vmse = parse_losses(args.log)
            ax.plot(it, pce, "o-", label="policy CE", color="#3b6")
            ax.set_xlabel("iteration"); ax.set_ylabel("policy CE", color="#3b6")
            ax2 = ax.twinx()
            ax2.plot(it, vmse, "s--", label="value MSE", color="#c63")
            ax2.set_ylabel("value MSE", color="#c63")
            ax.set_title("Training loss per iteration")
        else:
            it, wr, av = eval_strength(
                args.run_dir, args.baseline, args.eval_games, args.eval_sims)
            ax.axhline(25, ls=":", color="#999", label="random (25%)")
            ax.plot(it, wr, "o-", color="#36c", label="AZ win-rate")
            ax.set_xlabel("iteration"); ax.set_ylabel("win-rate vs " + args.baseline + " (%)")
            ax2 = ax.twinx()
            ax2.plot(it, av, "s--", color="#c63", label="avg score")
            ax2.set_ylabel("avg score", color="#c63")
            ax.set_ylim(0, 100)
            ax.legend(loc="upper left")
            ax.set_title(f"Strength vs 3x {args.baseline}")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
