"""Quick diagnostic: time one MCTS move from a fresh 4-player game state."""
import time

import _framework_path  # noqa: F401
from model import GameState  # noqa: E402

from agents.mcts import MCTSPlayer, fast_heuristic_rollout, random_rollout  # noqa: E402
from agents.sim import AzulSim  # noqa: E402


def time_one_move(iterations, rollout_fn, label):
    gs = GameState(4)
    # GameRunner.Run() normally calls StartRound for each player; we do it
    # manually here since we construct GameState directly.
    for p in gs.players:
        p.player_trace.StartRound()
    sim = AzulSim(gs, gs.first_player)
    moves = sim.legal_moves()
    print(f"  {label}: {len(moves)} legal moves at root")
    p = MCTSPlayer(gs.first_player, iterations=iterations, rollout_policy=rollout_fn)
    t0 = time.time()
    p.SelectMove(moves, gs)
    elapsed = time.time() - t0
    stats = p.last_stats
    print(f"    iterations={stats['iterations']}  time={elapsed:.2f}s  "
          f"iters/s={stats['iterations']/elapsed:.0f}")
    top3 = stats["candidates"][:3]
    for c in top3:
        print(f"      visits={c['visits']:>4}  avg_value={c['avg_value']:.3f}")


print("== random_rollout ==")
time_one_move(200, random_rollout, "200 iter")
time_one_move(1000, random_rollout, "1000 iter")

print("\n== fast_heuristic_rollout ==")
time_one_move(200, fast_heuristic_rollout, "200 iter")
time_one_move(1000, fast_heuristic_rollout, "1000 iter")
