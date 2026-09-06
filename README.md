---
title: Azul Coach
emoji: ðŸŽ²
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Play Azul against bots, with per-move MCTS coach hints.
---

Play the board game Azul (1 human vs 1-3 bots) with optional MCTS coach
suggestions before each of your moves. Source:
https://github.com/bltg85/azul-coach

Bundles a copy of the upstream framework
[michelleblom/AZUL](https://github.com/michelleblom/AZUL) under
`framework/` (GPLv3).


## Run locally

Install `requirements.txt`, then in PowerShell:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
python -c "from webapp import app; app.run(host='127.0.0.1', port=5055)"
```

Open http://127.0.0.1:5055. In a four-player game, use **Try opponent-aware
coach (experimental)** on your turn. It samples likely opponent moves and
possible future tile draws, with at most 128 complete simulations and an
8-second budget checked between simulations. The existing Aragon opponents
remain the default. Coach values are noisy simulated win shares, not calibrated
probabilities against humans; suggestions are ranked by search visits.

## Validate and benchmark

```powershell
python -m unittest discover -s tests -v
python bench_opponents.py --games 16 --sims 32 --workers 4
```

The benchmark sets BLAS thread limits itself. Use the limits above for tests
and the web server. See [experiment notes](experiments/opponent-search.md) for
results, reproducible commands, known limitations, and the next experiments.


For equal per-move wall-clock budgets against three time-adapted AZ bots:

```powershell
python bench_equal_time.py --games 48 --budgets .2 .5 --workers 4 --seed 91000
```

The checkpoint resumes with the same command and validates code/settings/weights.
This benchmark-only AZ adapter allocates time across sequential-halving phases;
it does not alter the default serving bot. See [equal-time results and protocol](experiments/equal-time.md).
