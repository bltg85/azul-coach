"""Generate self-play training data with PUCT + a net.

Each ply records: the canonical feature vector, the MCTS visit-count policy
(the policy target), and which seat was to move. After the game ends we turn
the final scores into a canonical value target per sample.

No torch needed — inference uses the numpy net. Run before training:

    python -m az.selfplay_az --games 50 --sims 120 --out data/sp1.npz
    python -m az.selfplay_az --games 50 --sims 120 --weights runs/az.npz --out data/sp2.npz
"""
import argparse
import os
import random
import sys
import time

import numpy as np

import _framework_path  # noqa: F401
from model import GameState  # noqa: E402

from az import NUM_PLAYERS  # noqa: E402
from az.actions import ACTION_SIZE  # noqa: E402
from az.encoder import FEATURE_SIZE  # noqa: E402
from az.encoder import encode  # noqa: E402
from az.mcts_az import az_search, policy_from_visits, _normalise_scores  # noqa: E402
from az.net import NumpyNet, random_numpynet  # noqa: E402
from agents.sim import AzulSim  # noqa: E402

# AlphaZero exploration schedule: sample moves from the visit distribution for
# the opening, then play greedily. The policy *target* is always the raw visit
# distribution (temperature 1) regardless of how we pick the move.
TEMP_MOVES = 20
DIRICHLET_ALPHA = 0.3


def play_game(evaluator, n_sims, c_puct=1.5, rng=None):
    rng = rng or np.random.default_rng()
    gs = GameState(NUM_PLAYERS)
    for p in gs.players:
        p.player_trace.StartRound()
    sim = AzulSim(gs, gs.first_player)

    feats_log, pi_log, seat_log = [], [], []
    ply = 0
    while not sim.terminal and ply < 400:
        root, visits = az_search(
            sim, evaluator, n_sims=n_sims, c_puct=c_puct,
            dirichlet_alpha=DIRICHLET_ALPHA,
        )
        if visits.sum() == 0:
            break
        pi_target = policy_from_visits(visits, temperature=1.0)
        feats_log.append(encode(sim.gs, sim.cur))
        pi_log.append(pi_target.astype(np.float32))
        seat_log.append(sim.cur)
        # pick the move to actually play
        temp = 1.0 if ply < TEMP_MOVES else 0.0
        pick_dist = policy_from_visits(visits, temperature=temp)
        a = int(rng.choice(ACTION_SIZE, p=pick_dist)) if temp > 0 else int(np.argmax(pick_dist))
        sim.apply(root.idx_to_move[a])
        ply += 1

    n = len(sim.gs.players)
    norm_final = _normalise_scores(sim.scores())
    X = np.asarray(feats_log, dtype=np.float32)
    P = np.asarray(pi_log, dtype=np.float32)
    V = np.zeros((len(seat_log), NUM_PLAYERS), dtype=np.float32)
    for i, s in enumerate(seat_log):
        for k in range(NUM_PLAYERS):
            V[i, k] = norm_final[(s + k) % n]
    return X, P, V, sim.scores()


# ---------------------------------------------------------------------------
# Parallel self-play (multiprocessing). Workers must be module-level so they
# pickle under Windows' spawn start method.
# ---------------------------------------------------------------------------

_WORKER = {}


def _init_worker(weights_path, net_seed, sims, c_puct):
    net = NumpyNet.load(weights_path) if weights_path else random_numpynet(net_seed)
    _WORKER["net"] = net
    _WORKER["sims"] = sims
    _WORKER["c_puct"] = c_puct


def _play_one(game_seed):
    rng = np.random.default_rng(game_seed)
    return play_game(_WORKER["net"].forward, _WORKER["sims"], _WORKER["c_puct"], rng)


def generate_parallel(weights, sims, c_puct, games, seed, workers):
    """Play `games` self-play games across `workers` processes. Returns
    concatenated (X, P, V) plus a list of final scores."""
    from multiprocessing import Pool

    seeds = [seed * 100003 + i for i in range(games)]
    Xs, Ps, Vs, all_scores = [], [], [], []
    t0 = time.time()
    with Pool(workers, initializer=_init_worker,
              initargs=(weights, seed, sims, c_puct)) as pool:
        done = 0
        for X, P, V, scores in pool.imap_unordered(_play_one, seeds):
            Xs.append(X); Ps.append(P); Vs.append(V); all_scores.append(scores)
            done += 1
            if done % 5 == 0 or done == games:
                n = sum(len(x) for x in Xs)
                print(f"  {done}/{games} games  samples={n}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
    return (np.concatenate(Xs), np.concatenate(Ps), np.concatenate(Vs), all_scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--sims", type=int, default=120)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--weights", default=None, help="NumpyNet .npz; random if omitted")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel self-play processes (1 = sequential)")
    args = ap.parse_args()

    t0 = time.time()
    if args.workers > 1:
        X, P, V, _ = generate_parallel(
            args.weights, args.sims, args.c_puct, args.games, args.seed, args.workers)
    else:
        net = NumpyNet.load(args.weights) if args.weights else random_numpynet(seed=args.seed)
        rng = np.random.default_rng(args.seed)
        random.seed(args.seed)
        Xs, Ps, Vs = [], [], []
        for g in range(args.games):
            X, P, V, scores = play_game(net.forward, args.sims, args.c_puct, rng)
            Xs.append(X); Ps.append(P); Vs.append(V)
            if (g + 1) % 5 == 0 or args.games <= 10:
                print(f"  game {g+1}/{args.games}  samples={sum(len(x) for x in Xs)}  "
                      f"({time.time()-t0:.0f}s)  last scores={scores}", flush=True)
        X = np.concatenate(Xs); P = np.concatenate(Ps); V = np.concatenate(Vs)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, X=X, P=P, V=V)
    print(f"Saved {len(X)} samples to {args.out} "
          f"(X{X.shape} P{P.shape} V{V.shape}) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
