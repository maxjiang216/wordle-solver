# Model improvement notes (paused 2026-10-17)

Exploratory work on the empirical prior. Nothing below is implemented yet.

## Findings

### Repeats (27 so far, as of 2026-10-16)
- 23/27 were first used in puzzles #0-199 (~8x the uniform rate); 2 from #200-299;
  only 2 later (SMILE #830, SANDY #1226).
- Minimum gap since last use: 524 days.
- Timing ~Poisson: mean 9.6 days between repeats (~10% of days), no weekday pattern.
- No visible ordering within the early words.
- Current model uses linear "years since last use" (x6.5/yr): gives early words ~20x,
  data says ~40x, and can't express the gap rule.

### Exclusion lists (independent sources, stratified permutation test by Zipf bin)
Sources: LDNOOBW `en` list, dominictarr/random-name first-names.txt, mledoze/countries.

| List | In orig list | Used | Expected | p |
|---|---|---|---|---|
| Offensive (LDNOOBW) | 6 | 0 | 4.7 | 0.0001 |
| First names | 78 | 60 | 64.0 | 0.14 (no effect) |
| Countries/capitals/demonyms | 3 | 1 | 2.6 | 0.046 (n too small) |

Unused offensive: FECAL GROPE HORNY KINKY SEMEN SPUNK. Names alone are not avoided
(CANDY, ROBIN, HAZEL used); unused names are mostly sexual/slur-like (FANNY, WILLY, GYPSY).
Rate-ratio estimate for offensive: ~x0.1 (0.5 pseudo-count / 4.7), 95% upper bound ~x0.64.

### Part of speech (WordNet 3.0 index files, dominant POS by tagged sense count)
- Original list: verbs x1.08 (p<0.001), nouns/adjectives/adverbs ~x1.0.
- Off-list picks: no POS effect (all p>0.4). "-Y" words 19% of picks = 19% of pool.
- Only signal: "not in WordNet" x0.70 (orig, p<0.001), x0.48 (off-list, p=0.05);
  overlaps the dict_lower feature.

## Candidate changes to test
1. Repeat rule: ~85% of REP mass uniform over early words (#0-199), rest over other
   words with gap >= ~500 days, x0.01 floor below 500 days (soft, never zero).
2. Offensive words x0.1 in ORIG and OFF.
3. in_wordnet feature.
4. Safety floor: ~1% mass uniform over all valid words.
5. Part of speech: dropped.

## Backtest speedup plan (do before testing)
Profiled: design rows 2.4s + Newton ~7s per fit, x22 refits x variants;
category_probs 11ms/day (recomputes history); predict 2.6ms/day.
1. Build per-day design rows once; refits use a prefix.
2. Warm-start Newton from previous month's theta (1-2 steps instead of ~8).
3. Incremental category counts.
4. One process per variant (20 cores).
5. Select features on prior log-loss (Part 1) only; solver backtest once on the winner.
6. Cache orig-list/classic solver baselines; use the warm opener search in the solver backtest.
Target: ~1-2 min per variant comparison, ~5 min final solver check (from 25+ min).

Evaluation: paired per-day bit differences with SE; select on 2025-01..2026-03,
confirm once on held-out 2026-04..2026-10. Rare-event changes (repeats, offensive)
will have wide error bars.

---------------------------------------------------------------------------------
# Update 2026-10-22: speedups done, features tested, daily prior changed

## Backtest speedups (scripts/empirical_prior.py, scripts/backtest_empirical_prior.py)
- `DayData`: per-day design rows built once, stacked per category; fit = segment softmax
  (np.*.reduceat) on any prefix; `fit(..., data, theta0)` warm-starts. Coefficients identical
  to the old loop (max diff 7e-15).
- `category_sequence` + `category_prob_table`: category probs O(1) per day (identical).
- Backtest: one forked process per variant, solver parallel over months, solver results
  cached in cache/solver_bt/ keyed by a hash of the daily priors. `--jobs` (default 4),
  `--variants candidates`, `--end`, `--out`; paired diffs vs `full` printed in Part 1 and 2.
- Full prior+solver backtest: 25+ min -> ~3 min at 4 workers (~1.5GB peak per worker).
  Part 1 output on the old window matched the old report exactly.
- Machine: 20 cores, 15GB RAM. Keep <=10 workers; 20 workers froze it.

## Feature results (Part 1 = bits/day vs full, negative = better; select 2025-01..2026-03,
## held-out 2026-04..2026-10-21)
| variant | select | held-out | solver (659 days, guesses/day vs full) |
|---|---|---|---|
| +wordnet (fitted) | -0.001 +-0.012 | - | dropped |
| +offensive x0.1 (fixed) | -0.010 | -0.016 | 0.000 +-0.016 (last 90 days) |
| +floor1% | -0.048 +-0.056 | -0.008 +-0.018 | +0.011 +-0.011 (last 90 days) |
| +rep-rule (85% uniform over first 200 puzzles, x0.01 < 500d) | -0.010 | -0.055 +-0.045 | **+0.056 +-0.024 worse** (last 90 days) |
| +min-gap (age/zipf logit kept, x0.01 if used < 500 days ago) | -0.003 +-0.002 | 0.000 | -0.002 +-0.002 |
| +gap+floor+off (= DAILY_SETTINGS) | -0.061 +-0.056 | -0.024 +-0.018 | -0.006 +-0.006 (532 vs 530 days <=3) |

Why rep-rule lost: a flat 85% over ~200 early words put weight on plausible used words that
matched the feedback, and the solver guessed them (TRIBE: SALET TRUCE TRIBE vs SALET TRIBE;
same for PUTTY, TWEEN, BEIGE, where p(answer) was identical). The hard #200 cutoff also cut
FOCUS (#267) 24x. The smooth age term already handles "time since last use".
Letter identity (Zipf-stratified permutation test): Y x0.92, S x0.95, E x1.03 in the
original list, nothing in off-list picks. Too small; not a feature.

## Changed for ./daily
- `DAILY_SETTINGS` in scripts/empirical_prior.py = min-gap + offensive x0.1 + 1% floor;
  used by `main()` (so ./daily). Model cache key includes the settings -> auto refit.
- New data: data/offensive_words.txt (LDNOOBW en, 29 words), data/wordnet_words.txt
  (WordNet 3.0 lemmas, 4278 words).
- src/weighted_three_solver.cpp: `--threads N` (default still all cores; ./daily unchanged).

## TODO
Run the weighted-three backtest (the C++ solver ./daily uses), full vs daily. Interrupted:
at 5 threads it is ~12-20 min per prior x 4 priors. To run faster:
    nice -n 19 python3 scripts/backtest_weighted_three.py --threads 10
(orig-list/classic don't depend on the change; could skip them to save half the time.)
Revert DAILY_SETTINGS to {} if `daily` loses to `full` there.
Nothing is committed yet.

## Result of the C++ weighted-three backtest (2026-10-22, 659 days) - DAILY_SETTINGS kept
daily vs full: P(<=3) 83.9% vs 82.9% (+1.1 pts, SE 0.8), mean guesses 3.200 vs 3.209 (-0.009, SE 0.011).
No game over 6 for either. Gain is within noise but consistent with the Python backtest.
Solver speedups (same day, byte-identical per-day results): exact early exit in the opener search
(145-149 of 150 candidates dropped early), seed opener, per-prior result cache, reuse of the
opener when the prior is unchanged. Full 3-prior run: 346s wall at 5 threads, 0.84GB RSS.
