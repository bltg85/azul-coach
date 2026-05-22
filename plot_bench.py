"""Read bench_results.json and render two charts:

  1) Win rate per bot type vs 3x heuristic (bar chart)
  2) MCTS strength vs iteration budget (line chart)
"""
import argparse
import json

import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="bench_results.json")
    ap.add_argument("--out-prefix", default="bench")
    args = ap.parse_args()

    with open(args.in_path) as f:
        data = json.load(f)
    rows = data["results"]
    n_games = data["games_per_config"]

    # ---- Graph 1: win rate per bot type ----
    labels = [r["label"] for r in rows]
    win_pct = [r["win_rate"] * 100 for r in rows]
    avg_score = [r["subject_avg"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    colors = [
        "#bbbbbb" if l in ("random", "naive") else
        "#888888" if l == "heuristic" else
        "#3b7dd8"  # MCTS variants
        for l in labels
    ]
    bars = ax1.bar(labels, win_pct, color=colors, edgecolor="black", linewidth=0.5)
    ax1.axhline(25, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
    ax1.text(len(labels) - 0.5, 26, "  random baseline (25%)", fontsize=8, va="bottom", ha="right")
    ax1.set_ylabel("Wins vs 3x heuristic (%)")
    ax1.set_ylim(0, max(100, max(win_pct) + 10))
    ax1.set_title(f"Bot strength (4-player Azul, {n_games} games per config)")
    for bar, pct in zip(bars, win_pct):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{pct:.0f}%", ha="center", va="bottom", fontsize=9)
    # Annotate avg score under each bar
    for i, (label, avg) in enumerate(zip(labels, avg_score)):
        ax1.text(i, -7, f"avg {avg:.0f}", ha="center", va="top", fontsize=8, color="#555")
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    plt.tight_layout()
    p1 = f"{args.out_prefix}_strength.png"
    plt.savefig(p1, dpi=130)
    plt.close()
    print(f"wrote {p1}")

    # ---- Graph 2: MCTS strength vs iterations ----
    mcts_rows = [r for r in rows if r["subject_spec"].startswith("mcts:")]
    if mcts_rows:
        iters = [int(r["subject_spec"].split(":")[1]) for r in mcts_rows]
        win_pct_m = [r["win_rate"] * 100 for r in mcts_rows]
        avg_m = [r["subject_avg"] for r in mcts_rows]
        elapsed_per_game = [r["elapsed_s"] / r["games"] for r in mcts_rows]

        # baseline for reference (heuristic-vs-heuristic = ~25%)
        baseline_rows = [r for r in rows if r["label"] == "heuristic"]
        baseline_pct = baseline_rows[0]["win_rate"] * 100 if baseline_rows else 25.0

        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(iters, win_pct_m, marker="o", linewidth=2, color="#3b7dd8",
                label="MCTS win % vs 3x heuristic")
        ax.axhline(baseline_pct, color="#888", linestyle="--", linewidth=1,
                   label=f"heuristic baseline ({baseline_pct:.0f}%)")
        ax.set_xscale("log")
        ax.set_xlabel("MCTS iterations per move (log scale)")
        ax.set_ylabel("Win rate (%)")
        ax.set_title(f"MCTS strength vs compute ({n_games} games per point)")
        ax.set_ylim(0, max(60, max(win_pct_m) + 10))
        ax.grid(True, which="both", alpha=0.25)

        # annotate with avg score and per-move time
        for x, y, avg, t in zip(iters, win_pct_m, avg_m, elapsed_per_game):
            ax.annotate(f"avg {avg:.0f}\n{t:.0f}s/game",
                        xy=(x, y), xytext=(8, 8), textcoords="offset points",
                        fontsize=8, color="#333")

        ax.legend(loc="lower right")
        plt.tight_layout()
        p2 = f"{args.out_prefix}_mcts_curve.png"
        plt.savefig(p2, dpi=130)
        plt.close()
        print(f"wrote {p2}")


if __name__ == "__main__":
    main()
