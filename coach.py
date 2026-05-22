"""Coach CLI: given a game state, show MCTS' top recommendations.

For now this generates a random 4-player game and plays out by asking the
coach for advice each turn (humans/naive opponents). The point is to
demonstrate the per-move output format that a real coach would show.

Usage:
    python coach.py                    # demo run: coach plays as player 0
    python coach.py --iter 1000        # more search per move
    python coach.py --players 3        # 3-player game
"""
import argparse
import time

import _framework_path  # noqa: F401
from model import GameState  # noqa: E402
from utils import MoveToString  # noqa: E402

from agents.heuristic import HeuristicPlayer  # noqa: E402
from agents.mcts import MCTSPlayer  # noqa: E402
from agents.sim import AzulSim  # noqa: E402


def show_recommendation(coach, sim, top=5):
    moves = sim.legal_moves()
    print(f"\n--- Player {sim.cur}'s turn — {len(moves)} legal moves ---")

    t0 = time.time()
    chosen = coach.SelectMove(moves, sim.gs)
    elapsed = time.time() - t0

    stats = coach.last_stats
    print(f"MCTS: {stats['iterations']} iterations in {elapsed:.1f}s "
          f"({stats['root_visits']} root visits)\n")
    print(f"  Top {min(top, len(stats['candidates']))} moves:")
    for i, c in enumerate(stats["candidates"][:top]):
        marker = ">>" if i == 0 else "  "
        # Full move description on multiple lines; indent continuations.
        lines = MoveToString(coach.id, c["move"]).split("\n")
        first = lines[0].replace(f"Player {coach.id} takes ", "")
        header = f"  {marker}  visits={c['visits']:>4}  value={c['avg_value']:.3f}  {first}"
        print(header)
        for cont in lines[1:]:
            print(" " * 38 + cont.strip())
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=int, default=4)
    ap.add_argument("--iter", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-turns", type=int, default=8, help="show coach for first N of its turns")
    args = ap.parse_args()

    import random
    random.seed(args.seed)

    gs = GameState(args.players)
    for p in gs.players:
        p.player_trace.StartRound()
    sim = AzulSim(gs, gs.first_player)

    coach = MCTSPlayer(_id=0, iterations=args.iter)
    bots = [HeuristicPlayer(i) for i in range(args.players)]

    shown = 0
    while not sim.terminal and shown < args.max_turns:
        if sim.cur == 0:
            move = show_recommendation(coach, sim)
            shown += 1
        else:
            moves = sim.legal_moves()
            move = bots[sim.cur].SelectMove(moves, sim.gs)
        sim.apply(move)

    print("\n" + "=" * 60)
    print("Demo ended.")
    print(f"Current scores: {sim.scores()}")
    print(f"Terminal: {sim.terminal}")


if __name__ == "__main__":
    main()
