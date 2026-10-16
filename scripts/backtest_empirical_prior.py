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
The solver caches a 14855x14855 feedback matrix (~220MB) in cache/patterns.npy.
"""

import argparse
import datetime as dt
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from empirical_prior import CATS, History, PriorModel, WordTable, load_answers  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_GREEN = 242
SCORE_CAP = 2000  # candidates used for scoring a guess (by prior weight)

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


VARIANTS = {
    'classic': None,
    'cats-only': dict(use_features=False, use_lag=False),
    'no-lag': dict(use_lag=False),
    'full': dict(),
    'full-w365': dict(fit_half_life=365.0),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--start', default='2025-01-01')
    ap.add_argument('--no-solver', action='store_true')
    ap.add_argument('--solver-variants', default='classic,full')
    ap.add_argument('--half-lives', default='30,60,120,0', help='category half-life sensitivity for "full" (0 = no decay)')
    args = ap.parse_args()
    start = dt.date.fromisoformat(args.start)

    table = WordTable()
    answers = load_answers()
    test = [i for i, (d, _) in enumerate(answers) if d >= start]
    emit(f'Walk-forward backtest, {answers[test[0]][0]} .. {answers[test[-1]][0]} ({len(test)} days)')
    emit('Model refitted at the start of each month on prior answers only.')
    emit()

    variants = dict(VARIANTS)
    for hl in args.half_lives.split(','):
        if hl and float(hl) != 60:
            variants[f'full-hl{hl}'] = dict(half_life=float(hl) or None)

    # priors[name][k] = probability vector for test day k (kept only as needed)
    bits = {v: [] for v in variants}
    ranks = {v: [] for v in variants}
    cat_of_day = []
    keep_priors = set(args.solver_variants.split(',')) if not args.no_solver else set()
    saved = {v: [] for v in keep_priors}
    models, fitted_month = {}, None
    coef_log = []

    hist = History(table)
    for i, (d, w) in enumerate(answers):
        if i >= test[0] and fitted_month != (d.year, d.month):
            fitted_month = (d.year, d.month)
            t0 = time.time()
            for v, kw in variants.items():
                if kw is not None:
                    models[v] = PriorModel(table, **kw).fit(answers[:i])
            coef_log.append((d, models['full'].describe()))
            print(f'  refit {d} ({time.time() - t0:.1f}s)', file=sys.stderr, flush=True)
        if i >= test[0]:
            cat_of_day.append(hist.category_of(w))
            ai = table.index[w]
            for v, kw in variants.items():
                if kw is None:
                    p = classic_prior(table, hist)
                else:
                    m = models[v]
                    p = m.predict(hist, d, m.category_probs(answers, i))
                bits[v].append(-np.log2(p[ai]))
                # Expected rank under random tie-breaking (uniform priors have big ties).
                ranks[v].append((p > p[ai]).sum() + ((p == p[ai]).sum() + 1) / 2)
                if v in saved:
                    saved[v].append(p.astype(np.float32))
        hist.push(d, w)

    cat_of_day = np.array(cat_of_day)
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
    emit()
    emit('Reference: uniform over all 14855 valid words = 13.86 bits; over 2309 original = 11.17 bits.')
    emit()
    emit('Fitted "full" coefficients at first and last refit:')
    for d, desc in (coef_log[0], coef_log[-1]):
        emit(f' refit {d}:')
        emit(desc)

    if not args.no_solver:
        emit()
        emit('=' * 78)
        emit('PART 2: SOLVER BACKTEST (same expected-cost solver; only the prior differs)')
        emit('=' * 78)
        M = feedback_matrix(table)
        results = {}
        for v in saved:
            t0 = time.time()
            turns = []
            opener, opener_month = None, None
            for k, i in enumerate(test):
                d, w = answers[i]
                p = saved[v][k].astype(np.float64)
                if opener_month != (d.year, d.month):  # opener recomputed monthly from that day's prior
                    opener_month = (d.year, d.month)
                    cand = np.flatnonzero(p > 0)
                    opener = best_guess(M, cand, p[cand])
                    print(f'  {v} opener {d}: {table.words[opener]}', file=sys.stderr, flush=True)
                turns.append(play(M, p, table.index[w], opener))
            results[v] = np.array(turns)
            print(f'  solver {v} done in {time.time() - t0:.0f}s', file=sys.stderr, flush=True)
        for ename, em in eras:
            emit(f'-- {ename} ({em.sum()} days)')
            emit(f'  {"prior":12s} {"mean":>6s} {"fail>6":>7s} ' + ' '.join(f'{k:>5d}' for k in range(1, 8)) + '   per-category mean')
            for v, tr in results.items():
                t = tr[em]
                dist = ' '.join(f'{(t == k).sum() if k < 7 else (t >= 7).sum():5d}' for k in range(1, 8))
                percat = ', '.join(f'{CATS[c]}={t[cat_of_day[em] == c].mean():.2f}'
                              for c in range(3) if (cat_of_day[em] == c).any())
                emit(f'  {v:12s} {t.mean():6.3f} {(t > 6).sum():7d} {dist}   {percat}')
        if 'classic' in results and 'full' in results:
            diff = results['classic'] - results['full']
            se = diff.std(ddof=1) / np.sqrt(len(diff))
            emit()
            emit(f'classic - full: {diff.mean():+.3f} guesses/day (paired SE {se:.3f});'
                 f' full better on {(diff > 0).sum()} days, worse on {(diff < 0).sum()}')

    path = os.path.join(ROOT, 'reports', 'empirical_prior_backtest.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(out_lines) + '\n')
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
