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

All 96 games completed. [Raw results](results/equal-time.json) include every move timing.

| Budget per move | New bot outright wins | Shared firsts | Win share | Approximate 95% seed-block interval |
| --- | ---: | ---: | ---: | --- |
| 0.2 s | 5/48 | 0 | 10.4% | 4.2% to 16.7% |
| 0.5 s | 6/48 | 0 | 12.5% | 6.2% to 18.8% |

Measured search time excludes forced moves:

| Budget | Method | Mean | p95 | Maximum | Mean completed simulations |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.2 s | learned | 0.2071 s | 0.2202 s | 0.2585 s | 47 |
| 0.2 s | az | 0.2047 s | 0.2075 s | 0.2227 s | 1098 |
| 0.5 s | learned | 0.5074 s | 0.5209 s | 0.5369 s | 125 |
| 0.5 s | az | 0.5110 s | 0.5204 s | 0.5688 s | 3027 |

Simulation counts are not comparable units of work: learned search runs complete
rollouts; AZ typically evaluates one network leaf per tree simulation.

| Budget | Seat 0 win share | Seat 1 | Seat 2 | Seat 3 |
| --- | ---: | ---: | ---: | ---: |
| 0.2 s | 8.3% | 8.3% | 16.7% | 8.3% |
| 0.5 s | 25.0% | 8.3% | 16.7% | 0.0% |

## Validation

22 tests pass, including deadline exhaustion during setup, full budget use for
non-power-of-two candidate counts, hidden-order/RNG invariance with a controlled
clock, and block-bootstrap/timing aggregation. The default serving AZ and
training entry points are unchanged. Experimental coach deadlines now include
root preparation as well as simulations.


## Conclusion and next decision

Keep Aragon as the default bot. The new rollout method won 5/48 (10.4%) and
6/48 (12.5%), below the symmetric 25% reference at both tested budgets.
The measured mean times differ by only 1.2% at 0.2 seconds and 0.7% at 0.5
seconds, so unused computation does not explain the gap in this run.
The approximate seed-block intervals also sit below 25%, although twelve
seed blocks and one opponent family do not establish universal superiority.

The previous 13/32 result at unequal compute is not a demonstrated improvement
under equal time. Seeds and the baseline's search schedule also changed, so
these experiments do not identify compute allocation as the sole cause.

This does not refute opponent prediction. At 0.5 seconds the new method completes
about 125 full rollouts per decision, while AZ completes about 3,027 shorter
network-backed simulations. These counts describe the cost difference; they
are not a direct measure of useful search. The algorithms also differ in
value objective and handling of hidden draws.

The next implementation experiment should avoid full-game rollouts at every
leaf: test an inexpensive value estimate while keeping the same opponent
model, then compare against a version without that model at equal time. A
terminal-win value head is one candidate, but its training data/search must
first pass the hidden-information audit. Do not scale training or replace the
default bot on the evidence currently available.
