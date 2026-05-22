"""Interactive play: you're player 0, the rest are bots.

Usage:
    python play.py                              # 3 heuristic bots, no coach hints
    python play.py --bots heuristic,heuristic,mcts:300
    python play.py --coach                      # show MCTS top-3 on your turn
    python play.py --coach --iter 1000          # stronger coach
    python play.py --seed 42                    # reproducible game
"""
import argparse
import random as _random
import time

import _framework_path  # noqa: F401
from iplayer import InteractivePlayer  # noqa: E402
from model import GameRunner, Player  # noqa: E402
from naive_player import NaivePlayer  # noqa: E402
from utils import MoveToString  # noqa: E402

from agents.heuristic import HeuristicPlayer  # noqa: E402
from agents.mcts import MCTSPlayer  # noqa: E402


def make_bot(spec, pid):
    spec = spec.strip().lower()
    if spec == "heuristic":
        return HeuristicPlayer(pid)
    if spec == "naive":
        return NaivePlayer(pid)
    if spec.startswith("mcts"):
        iters = 300
        if ":" in spec:
            iters = int(spec.split(":", 1)[1])
        return MCTSPlayer(pid, iterations=iters)
    raise SystemExit(f"unknown bot type: {spec!r}")


class CoachingPlayer(Player):
    """InteractivePlayer with MCTS top-3 suggestions printed before each prompt."""

    def __init__(self, _id, iterations=500):
        super().__init__(_id)
        self.interactive = InteractivePlayer(_id)
        self.mcts = MCTSPlayer(_id, iterations=iterations)

    def SelectMove(self, moves, game_state):
        print("\n" + "=" * 60)
        print(f"COACH is thinking ({self.mcts.iterations} MCTS iterations) ...")
        t0 = time.time()
        _ = self.mcts.SelectMove(moves, game_state)
        elapsed = time.time() - t0
        stats = self.mcts.last_stats
        print(f"(took {elapsed:.1f}s)")
        print(f"\nTop {min(3, len(stats['candidates']))} suggestions:")
        for i, c in enumerate(stats["candidates"][:3]):
            marker = ">>" if i == 0 else "  "
            print(f"  {marker} #{i+1}  visits={c['visits']:>4}  value={c['avg_value']:.3f}")
            for line in MoveToString(self.id, c["move"]).split("\n"):
                print(f"        {line.strip()}")
        print("=" * 60 + "\n")
        return self.interactive.SelectMove(moves, game_state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bots",
        default="heuristic,heuristic,heuristic",
        help="comma-separated bot specs for seats 1..N-1 (heuristic|naive|mcts[:iters])",
    )
    ap.add_argument("--coach", action="store_true",
                    help="show MCTS top-3 suggestions on your turn")
    ap.add_argument("--iter", type=int, default=500, help="MCTS iterations for coach")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else _random.randint(0, 2**31)
    bot_specs = [s.strip() for s in args.bots.split(",")]
    if not bot_specs:
        raise SystemExit("need at least 1 bot opponent")
    if len(bot_specs) > 3:
        raise SystemExit("max 3 bot opponents (4-player Azul)")

    if args.coach:
        you = CoachingPlayer(0, iterations=args.iter)
    else:
        you = InteractivePlayer(0)

    players = [you]
    for i, spec in enumerate(bot_specs, start=1):
        players.append(make_bot(spec, i))

    print(f"Starting {len(players)}-player game (seed={seed})")
    print(f"You are player 0. Opponents: {bot_specs}")
    if args.coach:
        print("Coach mode ON.\n")

    gr = GameRunner(players, seed)
    activity = gr.Run(True)

    print("\n" + "=" * 60)
    print("FINAL SCORES")
    for i in range(len(players)):
        tag = "(YOU)" if i == 0 else f"({bot_specs[i-1]})"
        print(f"  Player {i} {tag:<14} {activity[i][0]}")


if __name__ == "__main__":
    main()
