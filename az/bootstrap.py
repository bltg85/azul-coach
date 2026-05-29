"""Generate supervised bootstrap data from the heuristic player.

Pure self-play from a random net collapses (the net learns to dump tiles on
the floor — everyone ties at ~0, no value signal). AlphaGo's fix: imitate a
competent player first, *then* improve with self-play. Here the teacher is
the existing HeuristicPlayer.

Each ply records: the canonical features, a soft policy target (softmax of
the heuristic's move scores, aggregated by action index), and — after the
game — a placement-based value target. An epsilon fraction of moves is random
to diversify the visited states.

    python -m az.bootstrap --games 400 --workers 10 --out data/boot.npz
"""
import argparse
import os
import sys

import numpy as np

import _framework_path  # noqa: F401
from model import GameState  # noqa: E402

from az import NUM_PLAYERS  # noqa: E402
from az.actions import ACTION_SIZE, move_to_action  # noqa: E402
from az.encoder import encode  # noqa: E402
from az.mcts_az import placement_values  # noqa: E402
from agents.heuristic import HeuristicPlayer  # noqa: E402
from agents.sim import AzulSim  # noqa: E402

def play_game_heuristic(epsilon, rng, temp=0.5):
    gs = GameState(NUM_PLAYERS)
    for p in gs.players:
        p.player_trace.StartRound()
    sim = AzulSim(gs, gs.first_player)
    heur = HeuristicPlayer(0)

    feats_log, pi_log, seat_log = [], [], []
    ply = 0
    while not sim.terminal and ply < 400:
        legal = sim.legal_moves()
        if not legal:
            break
        heur.id = sim.cur
        scores = np.array([heur._evaluate(m, sim.gs) for m in legal], dtype=np.float64)
        z = scores / temp
        z -= z.max()
        w = np.exp(z)
        w /= w.sum()
        pi = np.zeros(ACTION_SIZE, dtype=np.float32)
        for m, prob in zip(legal, w):
            pi[move_to_action(m)] += prob
        feats_log.append(encode(sim.gs, sim.cur))
        pi_log.append(pi)
        seat_log.append(sim.cur)
        # choose move: epsilon random, else heuristic-best
        if rng.random() < epsilon:
            chosen = legal[rng.integers(len(legal))]
        else:
            chosen = legal[int(np.argmax(scores))]
        sim.apply(chosen)
        ply += 1

    n = len(sim.gs.players)
    pv = placement_values(sim.scores())
    X = np.asarray(feats_log, dtype=np.float32)
    P = np.asarray(pi_log, dtype=np.float32)
    V = np.zeros((len(seat_log), NUM_PLAYERS), dtype=np.float32)
    for i, s in enumerate(seat_log):
        for k in range(NUM_PLAYERS):
            V[i, k] = pv[(s + k) % n]
    return X, P, V, sim.scores()


# --- parallel workers (module-level for Windows spawn) ---
_W = {}


def _init(epsilon, temp):
    _W["epsilon"] = epsilon
    _W["temp"] = temp


def _play(game_seed):
    return play_game_heuristic(_W["epsilon"], np.random.default_rng(game_seed), _W["temp"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--epsilon", type=float, default=0.15)
    ap.add_argument("--temp", type=float, default=0.5,
                    help="softmax temperature on heuristic scores (lower = sharper)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import time
    t0 = time.time()
    seeds = [args.seed * 100003 + i for i in range(args.games)]
    Xs, Ps, Vs = [], [], []

    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers, initializer=_init, initargs=(args.epsilon, args.temp)) as pool:
            done = 0
            for X, P, V, sc in pool.imap_unordered(_play, seeds):
                Xs.append(X); Ps.append(P); Vs.append(V); done += 1
                if done % 25 == 0 or done == args.games:
                    print(f"  {done}/{args.games} games  "
                          f"samples={sum(len(x) for x in Xs)}  ({time.time()-t0:.0f}s)",
                          flush=True)
    else:
        rng = np.random.default_rng(args.seed)
        for i in range(args.games):
            X, P, V, sc = play_game_heuristic(args.epsilon, rng, args.temp)
            Xs.append(X); Ps.append(P); Vs.append(V)
            if (i + 1) % 25 == 0 or args.games <= 10:
                print(f"  {i+1}/{args.games}  samples={sum(len(x) for x in Xs)}  "
                      f"({time.time()-t0:.0f}s)", flush=True)

    X = np.concatenate(Xs); P = np.concatenate(Ps); V = np.concatenate(Vs)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, X=X, P=P, V=V)
    print(f"Saved {len(X)} samples to {args.out} "
          f"(X{X.shape} P{P.shape} V{V.shape}) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
