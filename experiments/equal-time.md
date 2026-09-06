# Equal-time comparison, 2026-09-07

## Protocol fixed before the main run

Run one opponent-aware bot against three AZ bots with the same per-move
wall-clock budget. Test 0.2 and 0.5 seconds on deal seeds 91000 through 91011,
rotating the tested bot through all four seats: 48 games per budget, 96 total.
Each game completes normally; shared first place splits win credit after the
completed-row tiebreak. Use the existing weights and four worker processes,
with one BLAS thread per worker. Warm inference before timing either bot.

```powershell
python bench_equal_time.py --games 48 --budgets .2 .5 --workers 4 --seed 91000 --out runs/equal-time.json
```

This is a fresh evaluation set, distinct from the previous fixed-simulation
pilot and follow-up. An initial main-run attempt was stopped after timing-only inspection revealed unused
AZ budget for non-power-of-two candidate counts; its outcomes are excluded.
The phase count was corrected before restarting. A four-game engineering smoke run at seed 90000 and 0.1
seconds was used only to check execution and time accounting; it is not included
in the results. The sample size and budgets were chosen before main-run outcomes.

## How time is allocated

The existing Aragon Gumbel entry point accepts simulation counts, not deadlines.
`az/timed_player.py` is a benchmark-only adaptation: same policy, value head,
top-16 root candidates, sequential-halving scores and non-root tree search.
Each halving phase receives an equal fraction of the remaining time; simulations
are allocated round-robin among surviving candidates. A deadline may interrupt
a partial round. This tests a time-adapted version, not the unchanged 80-simulation
serving configuration. It retains the legacy single hypothetical hidden world
per decision and rank-value objective.

Opponent-aware search uses its existing PUCT and terminal win-share rollouts,
with a fresh hidden world each simulation and the 85/15 response mixture.
Its deadline now also includes root preparation. Both methods complete an atomic
simulation already in progress, so measured time can exceed the nominal budget.
The rollout method has longer atomic work units. Report measured mean/p95/max,
not only requested budget. Single-action forced moves bypass both searches and
are excluded from timing summaries. No artificial sleeps are used to equalize
reported time. Bot RNG and game RNG remain separate.

## Interpretation

Games sharing a deal seed are correlated. Uncertainty is estimated by resampling
whole four-seat seed blocks (10,000 bootstrap samples), not individual games.
Report seat results as well as pooled win share. With only 12 blocks per budget,
intervals remain approximate. A 25% share is the symmetric four-player reference;
there is no separately measured all-AZ control in this run. Results describe one
bot family, not human or tournament strength, and do not isolate opponent-model
quality from the different value targets and search algorithms.

Wall-clock scheduling makes exact moves non-reproducible even with fixed seeds.
The report records settings, weights/code hashes, Python/platform and commit.
Checkpoints reject mismatched settings, code or weights when resuming.

## Results

Main run in progress. Raw results will be attached after all 96 games complete.
