#!/usr/bin/env python3
"""
Walk-forward backtest of the empirical answer prior (scripts/empirical_prior.py).

For every day in the test window the model is refitted at the start of each month
using only answers published before that month, so no future data leaks in.

Part 1 - prior quality: bits (-log2 p) assigned to the true answer, rank, top-k hit rates.
Part 2 - solver: an expected-cost solver plays every test day with each prior.
         The solver is identical across priors; only the prior differs.

Priors compared:
  classic     99% uniform over unused original-list words, 1% uniform over everything else
              (what a "remaining answers" solver assumes)
  cats-only   category mixture (ORIG/OFF/REP from recent counts), uniform within category
  no-lag      full feature model without the yesterday-lookalike features
  full        full model
  full-w365   full model, fit with 1-year half-life weighting on past answers

Usage:
    python3 scripts/backtest_empirical_prior.py                 # prior quality + solver
    python3 scripts/backtest_empirical_prior.py --no-solver
    python3 scripts/backtest_empirical_prior.py --start 2026-02-02

Output: reports/empirical_prior_backtest.txt
The solver caches a 14855x14855 feedback matrix (~220MB) in cache/patterns.npy, and each
variant's solver results in cache/solver_bt/ keyed by a hash of its daily priors.
Variants run in parallel (one process each); the solver runs in parallel over months.
"""

import argparse
import os

# One BLAS thread per process: parallelism comes from --jobs, not from numpy.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import datetime as dt
import hashlib
import multiprocessing as mp
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from empirical_prior import (CATS, DayData, History, PriorModel, WordTable, category_prob_table,  # noqa: E402
                             category_sequence, load_answers)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_GREEN = 242
SCORE_CAP = 2000  # candidates used for scoring a guess (by prior weight)
SOLVER_VERSION = 1  # bump when best_guess/play change (invalidates cache/solver_bt)

out_lines = []


def emit(s=''):
    print(s, flush=True)
    out_lines.append(s)


# ---------------------------------------------------------------- feedback matrix

def feedback_matrix(table):
    """M[a, g] = base-3 feedback code for guess g against answer a (2=green, 1=yellow)."""
    path = os.path.join(ROOT, 'cache', 'patterns.npy')
    if os.path.exists(path):
        M = np.load(path, mmap_mode='r')
        if M.shape == (table.n, table.n):
            return np.asarray(M)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    L = table.letters.astype(np.int16)
    n = table.n
    M = np.empty((n, n), dtype=np.uint8)
    pow3 = np.array([81, 27, 9, 3, 1], dtype=np.uint8)
    t0 = time.time()
    for s in range(0, n, 256):
        A = L[s:s + 256]                                  # (c,5) answers
        green = A[:, None, :] == L[None, :, :]            # (c,n,5)
        code = np.zeros(green.shape[:2], dtype=np.uint8)
        for i in range(5):
            gl = L[None, :, i]                            # guess letter at i
            avail = np.zeros(green.shape[:2], dtype=np.int8)
            for j in range(5):
                avail += (A[:, None, j] == gl) & ~green[:, :, j]
            prior = np.zeros_like(avail)
            for k in range(i):
                prior += (L[None, :, k] == gl) & ~green[:, :, k]
            yellow = ~green[:, :, i] & (avail > prior)
            code += pow3[i] * (2 * green[:, :, i] + yellow).astype(np.uint8)
        M[s:s + 256] = code
    np.save(path, M)
    emit(f'(built feedback matrix in {time.time() - t0:.0f}s)')
    return M


# ---------------------------------------------------------------- solver

def est_more(H):
    """Rough expected number of further guesses for a bucket with entropy H bits."""
    return 1.0 + 0.6 * np.log2(1.0 + H)


def best_guess(M, cand, w):
    order = np.argsort(-w)[:SCORE_CAP]
    S, ws = cand[order], w[order] / w[order].sum()
    if len(S) == 1:
        return S[0]
    n = M.shape[1]
    codes = M[S].astype(np.int64)                         # (m, n)
    flat = (codes + 243 * np.arange(n)[None, :]).ravel()
    Pb = np.bincount(flat, weights=np.repeat(ws, n), minlength=243 * n).reshape(n, 243)
    Q = np.bincount(flat, weights=np.repeat(ws * np.log2(ws), n), minlength=243 * n).reshape(n, 243)
    with np.errstate(divide='ignore', invalid='ignore'):
        Hb = np.where(Pb > 0, np.log2(Pb) - Q / Pb, 0.0)
    cost = Pb * est_more(np.maximum(Hb, 0))
    cost[:, ALL_GREEN] = 0
    total = 1.0 + cost.sum(1)
    return int(np.argmin(total))


def play(M, prior, answer, opener, max_turns=12):
    cand = np.flatnonzero(prior > 0)
    w = prior[cand]
    g = opener
    for turn in range(1, max_turns + 1):
        fb = M[answer, g]
        if fb == ALL_GREEN:
            return turn
        keep = M[cand, g] == fb
        cand, w = cand[keep], w[keep]
        g = best_guess(M, cand, w)
    return max_turns + 1


# ---------------------------------------------------------------- priors

def classic_prior(table, hist):
    p = np.full(table.n, 0.01)
    main = table.in_orig & (hist.last_used < 0)
    p[main] = 0.0
    p[~main] = 0.01 / (~main).sum()
    p[main] = 0.99 / main.sum()
    return p


OFFENSIVE = float(np.log(0.1))  # fixed x0.1 (0 of ~4.7 expected used; see model_improvement_notes)

VARIANTS = {
    'classic': None,
    'cats-only': dict(use_features=False, use_lag=False),
    'no-lag': dict(use_lag=False),
    'full': dict(),
    'full-w365': dict(fit_half_life=365.0),
    # candidate changes, each alone on top of "full", then all together
    '+wordnet': dict(fit=('wordnet',)),
    '+offensive': dict(fixed={'offensive': OFFENSIVE}),
    '+rep-rule': dict(rep_rule=True),
    '+floor1%': dict(floor=0.01),
    '+all': dict(fit=('wordnet',), fixed={'offensive': OFFENSIVE}, rep_rule=True, floor=0.01),
    # picked on 2025-01..2026-03 prior log-loss (wordnet dropped: no gain)
    '+chosen': dict(fixed={'offensive': OFFENSIVE}, rep_rule=True, floor=0.01),
    # no repeat yet within 524 days of last use: x0.01 below 500 days, age/zipf logit kept
    '+min-gap': dict(fixed={'recent': float(np.log(0.01))}),
    '+gap+floor+off': dict(fixed={'recent': float(np.log(0.01)), 'offensive': OFFENSIVE}, floor=0.01),
}
DEFAULT_VARIANTS = 'classic,cats-only,no-lag,full,full-w365'
CANDIDATES = 'full,+wordnet,+offensive,+rep-rule,+floor1%,+all,+chosen'


# Shared read-only state for forked workers.
G = {}


def run_prior(v):
    """Walk-forward prior for one variant: bits, ranks and (optionally) the daily priors."""
    table, answers, data, test, cats = G['table'], G['answers'], G['data'], G['test'], G['cats']
    kw = G['variants'][v]
    bits, ranks, saved, coef_log = [], [], [], []
    model, fitted_month, theta = None, None, None
    cp_table = None
    if kw is not None:
        model = PriorModel(table, **kw)
        cp_table = category_prob_table(cats, model.half_life, model.pseudo)
    hist = History(table)
    t0 = time.time()
    for i, (d, w) in enumerate(answers):
        if i >= test[0]:
            if model is not None and fitted_month != (d.year, d.month):
                fitted_month = (d.year, d.month)
                model.fit(answers[:i], data, theta0=theta)  # warm start from last month's fit
                theta = model.theta
                coef_log.append((d, model.describe()))
            p = classic_prior(table, hist) if model is None else model.predict(hist, d, cp_table[i])
            ai = table.index[w]
            bits.append(-np.log2(p[ai]))
            # Expected rank under random tie-breaking (uniform priors have big ties).
            ranks.append((p > p[ai]).sum() + ((p == p[ai]).sum() + 1) / 2)
            if v in G['keep']:
                saved.append(p.astype(np.float32))
        hist.push(d, w)
    print(f'  prior {v} done in {time.time() - t0:.0f}s', file=sys.stderr, flush=True)
    return v, np.array(bits), np.array(ranks), (np.array(saved) if saved else None), coef_log


def run_month(job):
    """Solver on one month of test days: opener from the first day's prior, then play each day."""
    v, ks = job
    M, table, answers, test, P = G['M'], G['table'], G['answers'], G['test'], G['saved'][v]
    p = P[ks[0]].astype(np.float64)
    cand = np.flatnonzero(p > 0)
    opener = best_guess(M, cand, p[cand])
    turns = [play(M, P[k].astype(np.float64), table.index[answers[test[k]][1]], opener) for k in ks]
    return v, ks, turns, table.words[opener]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--start', default='2025-01-01')
    ap.add_argument('--end', default=None, help='last test day (inclusive)')
    ap.add_argument('--no-solver', action='store_true')
    ap.add_argument('--variants', default=None,
                    help=f'comma list, or "candidates" for {CANDIDATES} (default: {DEFAULT_VARIANTS} + half-lives)')
    ap.add_argument('--out', default=os.path.join(ROOT, 'reports', 'empirical_prior_backtest.txt'))
    ap.add_argument('--solver-variants', default='classic,full')
    ap.add_argument('--half-lives', default='30,60,120,0', help='category half-life sensitivity for "full" (0 = no decay)')
    ap.add_argument('--jobs', type=int, default=4,
                    help='worker processes (~1GB RAM each; machine has 15GB: stay <=10 and leave RAM free)')
    args = ap.parse_args()
    start = dt.date.fromisoformat(args.start)
    t_start = time.time()

    table = WordTable()
    answers = load_answers()
    if args.end:
        answers = [a for a in answers if a[0] <= dt.date.fromisoformat(args.end)]
    test = [i for i, (d, _) in enumerate(answers) if d >= start]
    emit(f'Walk-forward backtest, {answers[test[0]][0]} .. {answers[test[-1]][0]} ({len(test)} days)')
    emit('Model refitted at the start of each month on prior answers only.')
    emit()

    variants = {v: VARIANTS[v] for v in DEFAULT_VARIANTS.split(',')}
    for hl in args.half_lives.split(','):
        if hl and float(hl) != 60:
            variants[f'full-hl{hl}'] = dict(half_life=float(hl) or None)
    if args.variants:
        names = CANDIDATES if args.variants == 'candidates' else args.variants
        variants = {v: {**variants, **VARIANTS}[v] for v in names.split(',')}

    keep = set(args.solver_variants.split(',')) & set(variants) if not args.no_solver else set()
    G.update(table=table, answers=answers, data=DayData(table, answers), test=test,
             cats=category_sequence(table, answers), variants=variants, keep=keep)
    ctx = mp.get_context('fork')
    with ctx.Pool(min(args.jobs, len(variants))) as pool:
        res = {r[0]: r[1:] for r in pool.imap_unordered(run_prior, list(variants))}
    bits = {v: res[v][0] for v in variants}
    ranks = {v: res[v][1] for v in variants}
    saved = {v: res[v][2] for v in keep}
    coef_log = res['full'][3] if 'full' in res else next((r[3] for r in res.values() if r[3]), [])
    print(f'  priors done at {time.time() - t_start:.0f}s', file=sys.stderr, flush=True)

    cat_of_day = G['cats'][test]

    dates = [answers[i][0] for i in test]
    rep_start = next((dates[k] for k in range(len(test)) if cat_of_day[k] == 2), None)
    eras = [('all test days', np.ones(len(test), bool))]
    if rep_start:
        eras.append((f'since {rep_start}', np.array([x >= rep_start for x in dates])))
    eras.append(('last 90 days', np.arange(len(test)) >= len(test) - 90))

    emit('=' * 78)
    emit('PART 1: PRIOR QUALITY  (bits = -log2 p(actual answer); lower is better)')
    emit('=' * 78)
    for ename, em in eras:
        emit(f'-- {ename} ({em.sum()} days)')
        emit(f'  {"prior":12s} {"mean bits":>9s} {"median rank":>11s} {"top10":>6s} {"top100":>7s} {"top500":>7s}'
             + ''.join(f' {"bits|" + c:>10s}' for c in CATS))
        for v in variants:
            b, r = np.array(bits[v])[em], np.array(ranks[v])[em]
            percat = []
            for c in range(3):
                mm = cat_of_day[em] == c
                percat.append(f'{b[mm].mean():10.2f}' if mm.any() else f'{"-":>10s}')
            emit(f'  {v:12s} {b.mean():9.2f} {np.median(r):11.0f} {100 * (r <= 10).mean():5.1f}%'
                 f' {100 * (r <= 100).mean():6.1f}% {100 * (r <= 500).mean():6.1f}%' + ''.join(percat))
        emit(f'  category counts: ' + ', '.join(f'{c}={int((cat_of_day[em] == k).sum())}' for k, c in enumerate(CATS)))
        if 'full' in bits:
            emit('  paired vs full (bits/day, negative = better; +-1 SE):')
            for v in variants:
                if v == 'full':
                    continue
                d = (np.array(bits[v]) - np.array(bits['full']))[em]
                parts = [f'all {d.mean():+.3f} +-{d.std(ddof=1) / np.sqrt(len(d)):.3f}']
                for c in range(3):
                    dc = d[cat_of_day[em] == c]
                    if len(dc) > 1:
                        parts.append(f'{CATS[c]} {dc.mean():+.3f} +-{dc.std(ddof=1) / np.sqrt(len(dc)):.3f} (n={len(dc)})')
                emit(f'    {v:12s} ' + ', '.join(parts))
    emit()
    emit('Reference: uniform over all 14855 valid words = 13.86 bits; over 2309 original = 11.17 bits.')
    emit()
    if coef_log:
        emit('Fitted coefficients at first and last refit:')
        for d, desc in (coef_log[0], coef_log[-1]):
            emit(f' refit {d}:')
            emit(desc)

    if not args.no_solver:
        emit()
        emit('=' * 78)
        emit('PART 2: SOLVER BACKTEST (same expected-cost solver; only the prior differs)')
        emit('=' * 78)
        M = feedback_matrix(table)
        G.update(M=M, saved=saved)
        months = {}
        for k, i in enumerate(test):
            months.setdefault((answers[i][0].year, answers[i][0].month), []).append(k)
        cache_dir = os.path.join(ROOT, 'cache', 'solver_bt')
        os.makedirs(cache_dir, exist_ok=True)
        results, jobs, paths = {}, [], {}
        for v in saved:
            # Key: the exact daily priors, test answers and solver settings.
            h = hashlib.sha1(saved[v].tobytes())
            h.update(','.join(answers[i][1] for i in test).encode())
            h.update(f'{SOLVER_VERSION},{SCORE_CAP}'.encode())
            paths[v] = os.path.join(cache_dir, h.hexdigest()[:16] + '.npy')
            if os.path.exists(paths[v]):
                results[v] = np.load(paths[v])
                print(f'  solver {v}: cached', file=sys.stderr, flush=True)
            else:
                results[v] = np.zeros(len(test), dtype=np.int64)
                jobs += [(v, ks) for ks in months.values()]
        t0 = time.time()
        if jobs:
            with ctx.Pool(args.jobs) as pool:
                for v, ks, turns, op in pool.imap_unordered(run_month, jobs):
                    results[v][ks] = turns
            for v in saved:
                if not os.path.exists(paths[v]):
                    np.save(paths[v], results[v])
            print(f'  solver done in {time.time() - t0:.0f}s', file=sys.stderr, flush=True)
        for ename, em in eras:
            emit(f'-- {ename} ({em.sum()} days)')
            emit(f'  {"prior":12s} {"mean":>6s} {"fail>6":>7s} ' + ' '.join(f'{k:>5d}' for k in range(1, 8)) + '   per-category mean')
            for v, tr in results.items():
                t = tr[em]
                dist = ' '.join(f'{(t == k).sum() if k < 7 else (t >= 7).sum():5d}' for k in range(1, 8))
                percat = ', '.join(f'{CATS[c]}={t[cat_of_day[em] == c].mean():.2f}'
                              for c in range(3) if (cat_of_day[em] == c).any())
                emit(f'  {v:12s} {t.mean():6.3f} {(t > 6).sum():7d} {dist}   {percat}')
        if 'full' in results:
            emit()
            for v in results:
                if v == 'full':
                    continue
                diff = results[v] - results['full']
                se = diff.std(ddof=1) / np.sqrt(len(diff))
                emit(f'{v} - full: {diff.mean():+.3f} guesses/day (paired SE {se:.3f});'
                     f' {v} better on {(diff < 0).sum()} days, worse on {(diff > 0).sum()}')

    path = args.out
    with open(path, 'w') as f:
        f.write('\n'.join(out_lines) + '\n')
    print(f'\nwrote {path} ({time.time() - t_start:.0f}s total)')


if __name__ == '__main__':
    main()
