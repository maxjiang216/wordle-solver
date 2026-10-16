#!/usr/bin/env python3
"""
Walk-forward backtest of src/weighted_three_solver.cpp under different priors.

For each test day the prior is built from answers before that day only (the
empirical model is refitted monthly), written to cache/three_bt/, and the C++
solver replays every day (re-searching its opener each month).

Priors:
  orig-list  99.99% uniform over the 2309 original-list words, rest spread thinly
             (behaves like best_three_solver: answer list + fallback)
  classic    99% uniform over unused original-list words, 1% over everything else
  full       empirical model (scripts/empirical_prior.py)

Usage:
    make weighted_three_solver
    python3 scripts/backtest_weighted_three.py [--start 2025-01-01] [--max-first 150]

Output: reports/weighted_three_backtest.txt (per-day turns in cache/three_bt/results_*.csv)
"""

import argparse
import datetime as dt
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_empirical_prior import classic_prior  # noqa: E402
from empirical_prior import CATS, History, PriorModel, WordTable, load_answers  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, 'cache', 'three_bt')

out_lines = []


def emit(s=''):
    print(s, flush=True)
    out_lines.append(s)


def orig_list_prior(table):
    p = np.where(table.in_orig, 0.9999 / table.in_orig.sum(), 0.0001 / (~table.in_orig).sum())
    return p


def build_priors(table, answers, test):
    priors = {'orig-list': [], 'classic': [], 'full': []}
    cats = []
    hist = History(table)
    model, month = None, None
    for i, (d, w) in enumerate(answers):
        if i >= test[0]:
            if month != (d.year, d.month):
                month = (d.year, d.month)
                model = PriorModel(table).fit(answers[:i])
                print(f'  refit {d}', file=sys.stderr, flush=True)
            cats.append(hist.category_of(w))
            priors['orig-list'].append(orig_list_prior(table))
            priors['classic'].append(classic_prior(table, hist))
            priors['full'].append(model.predict(hist, d, model.category_probs(answers, i)))
        hist.push(d, w)
    return priors, np.array(cats)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--start', default='2025-01-01')
    ap.add_argument('--max-first', type=int, default=150)
    args = ap.parse_args()

    binary = os.path.join(ROOT, 'weighted_three_solver')
    if not os.path.exists(binary):
        sys.exit('build it first: make weighted_three_solver')
    table = WordTable()
    answers = load_answers()
    start = dt.date.fromisoformat(args.start)
    test = [i for i, (d, _) in enumerate(answers) if d >= start]
    dates = [answers[i][0] for i in test]

    os.makedirs(OUT_DIR, exist_ok=True)
    days_path = os.path.join(OUT_DIR, 'days.csv')
    with open(days_path, 'w') as f:
        f.write('date,answer\n')
        for i in test:
            f.write(f'{answers[i][0]},{answers[i][1]}\n')

    priors, cats = build_priors(table, answers, test)
    results = {}
    for name, ps in priors.items():
        bin_path = os.path.join(OUT_DIR, f'prior_{name}.bin')
        np.asarray(ps, dtype=np.float32).tofile(bin_path)
        t0 = time.time()
        proc = subprocess.run([binary, '--max-first', str(args.max_first), '--backtest', days_path, bin_path],
                              cwd=ROOT, capture_output=True, text=True, check=True)
        with open(os.path.join(OUT_DIR, f'results_{name}.csv'), 'w') as f:
            f.write('date,answer,turns,opener\n' + proc.stdout)
        rows = [line.split(',') for line in proc.stdout.strip().splitlines()]
        assert len(rows) == len(test), f'{name}: got {len(rows)} rows'
        results[name] = (np.array([int(r[2]) for r in rows]), [r[3] for r in rows])
        print(f'  {name} done in {time.time() - t0:.0f}s', file=sys.stderr, flush=True)

    emit(f'Weighted best-of-three solver, walk-forward {dates[0]} .. {dates[-1]} ({len(test)} days)')
    emit('Prior rebuilt from past answers only; opener re-searched monthly '
         f'(best of top {args.max_first} by bound).')
    rep_start = next((dates[k] for k in range(len(test)) if cats[k] == 2), None)
    eras = [('all test days', np.ones(len(test), bool))]
    if rep_start:
        eras.append((f'since {rep_start}', np.array([x >= rep_start for x in dates])))
    eras.append(('last 90 days', np.arange(len(test)) >= len(test) - 90))
    for ename, em in eras:
        emit()
        emit(f'-- {ename} ({em.sum()} days; ' + ', '.join(f'{c}={int((cats[em] == k).sum())}' for k, c in enumerate(CATS)) + ')')
        emit(f'  {"prior":10s} {"P<=2":>6s} {"P<=3":>6s} {"mean":>6s} {"fail>6":>6s}   '
             + ' '.join(f'{k:>4d}' for k in range(1, 8)) + '   P<=3 by category')
        for name, (t, _) in results.items():
            t = t[em]
            dist = ' '.join(f'{(t == k).sum() if k < 7 else (t >= 7).sum():4d}' for k in range(1, 8))
            bycat = ', '.join(f'{CATS[c]} {100 * (t[cats[em] == c] <= 3).mean():.0f}%'
                              for c in range(3) if (cats[em] == c).any())
            emit(f'  {name:10s} {100 * (t <= 2).mean():5.1f}% {100 * (t <= 3).mean():5.1f}% {t.mean():6.3f}'
                 f' {(t > 6).sum():6d}   {dist}   {bycat}')
    emit()
    for a, b in (('orig-list', 'full'), ('classic', 'full')):
        x, y = results[a][0], results[b][0]
        d3 = (y <= 3).astype(int) - (x <= 3).astype(int)
        emit(f'{b} vs {a}: P<=3 {100 * d3.mean():+.1f} pts (SE {100 * d3.std(ddof=1) / np.sqrt(len(d3)):.1f}),'
             f' mean guesses {(y - x).mean():+.3f} (SE {(y - x).std(ddof=1) / np.sqrt(len(x)):.3f})')
    emit()
    emit('Days needing 6+ guesses:')
    for name, (t, _) in results.items():
        bad = [f'{dates[k]} {answers[test[k]][1]} ({t[k]})' for k in np.flatnonzero(t >= 6)]
        emit(f'  {name:10s} ' + (', '.join(bad) if bad else 'none'))
    emit()
    emit('Monthly openers:')
    for name, (_, openers) in results.items():
        months, seen = [], None
        for d, o in zip(dates, openers):
            if (d.year, d.month) != seen:
                seen = (d.year, d.month)
                months.append(o.upper())
        emit(f'  {name:10s} ' + ' '.join(months))

    path = os.path.join(ROOT, 'reports', 'weighted_three_backtest.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(out_lines) + '\n')
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
