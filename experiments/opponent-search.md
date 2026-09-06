# Opponent-aware search experiment, 2026-09-07

## What changed

The new optional four-player coach uses the shipped network policy to predict
moves. Opponent turns sample a mixture: 85% of probability is restricted to the
three highest-prior legal actions and 15% uses the full prior. Predictions are
recomputed after each move. Own turns use PUCT. Leaves roll out to actual game
completion and optimize first place, splitting credit on a shared win after the
completed-row tiebreak. The existing rank-value head is not used as a win estimate.

Every simulation samples a new hidden bag/discard ordering and independent game
RNG. Search nodes distinguish public future factory layouts. Game RNG is isolated
from bot RNG; served AZ and legacy MCTS bots also sample hypothetical hidden
orders. Those legacy bots still search only one sampled world per decision.
Vendored framework import precedence was corrected for Windows spawned workers.

The UI adds an experimental coach button. It does not replace default opponents.
The simulator and UI now use the completed-row tiebreak when identifying winners.

## Evidence

All games use four players, shipped `az/weights/az_v1.npz`, and three AZ Gumbel
opponents with 80 simulations. Raw JSON includes weight hash, seed, seat, scores,
winners, per-move timings and settings. Rotate the tested bot through all seats.
The all-AZ control repeats an identical game for each scored seat; its 25% share
is therefore a sanity check by construction, not independent measured strength.

### Pilot: 4 deal seeds, 16 seat-rotated games per configuration

```powershell
python bench_opponents.py --games 16 --sims 32 --workers 4 --out runs/opponent-pilot.json
```

| Tested bot | Outright wins | Shared firsts | Mean win share | Mean seconds/move |
| --- | ---: | ---: | ---: | ---: |
| AZ control, 80 sims | 4/16 | 0 | 25.0% | 0.020 |
| Learned opponent model, 32 sims | 3/16 | 0 | 18.75% | 0.316 |
| Greedy opponent model, 32 sims | 2/16 | 0 | 12.5% | 0.300 |
| Uniform opponent model, 32 sims | 1/16 | 1 | 9.375% | 0.326 |

[Raw pilot results](results/opponent-pilot.json).

### Fresh-seed follow-up: 8 deal seeds, 32 seat-rotated games per configuration

```powershell
python bench_opponents.py --games 32 --sims 128 --workers 4 --seed 80000 --configs az learned --out runs/opponent-holdout.json
```

| Tested bot | Outright wins | Mean win share | Mean seconds/move | p95 seconds/move |
| --- | ---: | ---: | ---: | ---: |
| AZ control, 80 sims | 8/32 | 25.0% | 0.056 | 0.106 |
| Learned opponent model, 128 sims | 13/32 | 40.625% | 1.252 | 2.404 |

[Raw follow-up results](results/opponent-holdout.json).

These are exploratory, correlated seat-rotated samples against one opponent
family. Follow-up changes both seed and budget, so it does not isolate the
benefit of more simulations. Search methods are not matched on wall time, and
four concurrent worker processes affect latency. The new method is promising
at the larger budget but a strength improvement is not established.

## Does top-three prediction work?

In the pilot, the policy top three contained 92.22% of actual AZ search choices
across 360 positions; in the follow-up, 92.24% across 722 positions. Top one
contained 70.83% and 69.53%, respectively. Repeated control games are counted
only once per seed. These are predictions of a bot using the same network,
not evidence of human prediction accuracy or calibrated response probabilities.

## Validation and remaining limits

18 unit/integration tests cover hidden-order invariance for served bots, RNG
isolation, fresh sampling per simulation, public-state keys, terminal win targets,
tiebreaks, response distributions, coach routes/errors/stale results, and complete
2/3/4-player simulator games matching the original runner with 100-tile conservation.
The local browser was checked by requesting experimental advice in a real game.

No weights were trained. Direct legacy self-play search entry points are not
converted to this information-set approach and must be audited before another
training run. Existing uncommitted training experiments were kept separate from
this change. Full rollouts are costly; low-visit candidate values are very noisy.
The 8-second UI budget can overrun by one complete simulation.

## Next experiments

1. Match wall-clock budgets and use more unseen deal seeds, reporting results
   clustered by seed and by seat; compare against several opponent strengths.
2. Repeat learned/greedy/uniform ablations at the larger budget on the same deals.
3. With permitted human replay data, measure held-out top-k coverage and log loss;
   calibrate response probabilities by skill or player instead of assuming 85/15.
4. Only then consider training a terminal win-value head to replace expensive
   rollouts, evaluating search quality before scaling self-play training.
