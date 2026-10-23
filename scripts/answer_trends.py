#!/usr/bin/env python3
"""
Investigate how NYT Wordle answer selection has drifted from the original
curated answer list.

Sections:
  1. Repeat answers (when they started, rate, gap since first use)
  2. Answers outside the original list, by quarter
  3. Word frequency (wordfreq Zipf) of answers vs the candidate pools,
     and dictionary/proper-noun/inflection filters
  4. Consecutive-day letter correlation, with permutation tests

Inputs: data/wordle_answers.csv, data/wordle_words.txt, data/word_features.csv
        (build the latter with scripts/build_word_features.py)
Output: reports/answer_trends_report.txt (also printed)
"""

import csv
import os
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_PERM = 5000

out_lines = []


def emit(s=''):
    print(s)
    out_lines.append(s)


def load():
    rows = list(csv.DictReader(open(os.path.join(ROOT, 'data', 'wordle_answers.csv'))))
    feats = {r['word']: r for r in csv.DictReader(open(os.path.join(ROOT, 'data', 'word_features.csv')))}
    return rows, feats


def quarter(d):
    return f'{d[:4]}Q{(int(d[5:7]) - 1) // 3 + 1}'


def section_repeats_and_offlist(rows, feats):
    orig = {w for w, f in feats.items() if f['in_orig'] == '1'}
    emit('=' * 78)
    emit('1. REPEATS AND OFF-LIST ANSWERS')
    emit('=' * 78)
    first = {}
    by_q = defaultdict(Counter)
    repeats = []
    for r in rows:
        w, d, n = r['word'].upper(), r['date'], int(r['puzzle_number'])
        q = quarter(d)
        by_q[q]['n'] += 1
        if w in first:
            by_q[q]['repeat'] += 1
            repeats.append((d, n, w, first[w][0], n - first[w][1]))
        elif w not in orig:
            by_q[q]['offlist'] += 1
        else:
            by_q[q]['orig_new'] += 1
        first.setdefault(w, (d, n))

    emit(f'Answers: {len(rows)} ({rows[0]["date"]} .. {rows[-1]["date"]})')
    emit(f'Original-list words never used yet: {len(orig - set(first))} of {len(orig)}')
    emit()
    emit(f'{"quarter":8s} {"n":>4s} {"orig_new":>9s} {"offlist":>8s} {"repeat":>7s} {"off%":>6s} {"rep%":>6s}')
    for q in sorted(by_q):
        c = by_q[q]
        emit(f'{q:8s} {c["n"]:4d} {c["orig_new"]:9d} {c["offlist"]:8d} {c["repeat"]:7d} '
             f'{100 * c["offlist"] / c["n"]:6.1f} {100 * c["repeat"] / c["n"]:6.1f}')
    emit()
    emit(f'Repeats ({len(repeats)}); first repeat {repeats[0][0] if repeats else "-"}:')
    for d, n, w, fd, gap in repeats:
        emit(f'  {d} #{n:<5d} {w}  first used {fd}  gap {gap} days')
    if repeats:
        start = repeats[0][1]
        era = [r for r in rows if int(r['puzzle_number']) >= start]
        first_years = Counter(fd[:4] for _, _, _, fd, _ in repeats)
        # Where would uniformly random repeats have come from? Pool at the first repeat.
        pool_years = Counter(d[:4] for w, (d, n) in first.items() if n < start)
        tot = sum(pool_years.values())
        emit()
        emit(f'Repeat rate since {repeats[0][0]}: {len(repeats)}/{len(era)} = {100 * len(repeats) / len(era):.1f}%')
        emit('First-use year of repeated words vs share of all previously-used words:')
        for y in sorted(pool_years):
            emit(f'  {y}: repeats {first_years.get(y, 0):3d} ({100 * first_years.get(y, 0) / len(repeats):5.1f}%)'
                 f'   pool share {100 * pool_years[y] / tot:5.1f}%')
        emit(f'Min gap between uses: {min(g for *_, g in repeats)} days')


def section_frequency(rows, feats):
    emit()
    emit('=' * 78)
    emit('2. WORD FREQUENCY (wordfreq Zipf; 0 = unseen, ~3 = once per million words)')
    emit('=' * 78)
    used = [r['word'].upper() for r in rows]
    used_set = set(used)
    z = {w: float(f['zipf']) for w, f in feats.items()}
    orig = {w for w, f in feats.items() if f['in_orig'] == '1'}
    off = sorted({w for w in used if w not in orig})
    pool = [w for w in feats if w not in orig]

    def mean(ws):
        return np.mean([z[w] for w in ws])

    emit(f'Mean Zipf: orig-list used {mean(orig & used_set):.2f} | orig-list unused {mean(orig - used_set):.2f}'
         f' | off-list answers {mean(off):.2f} | non-orig valid guesses {mean(pool):.2f}')
    emit()
    emit('NYT appears to consume the original list most-familiar-first: the unused remainder is rarer.')
    bins = [0, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 9]
    emit(f'{"zipf bin":10s} {"orig used":>9s} {"unused":>7s} {"%unused":>8s}')
    for lo, hi in zip(bins, bins[1:]):
        a = sum(1 for w in orig & used_set if lo <= z[w] < hi)
        b = sum(1 for w in orig - used_set if lo <= z[w] < hi)
        emit(f'{lo:4.1f}-{hi:<4.1f}  {a:9d} {b:7d} {100 * b / max(a + b, 1):7.0f}%')

    emit()
    emit('Off-list pick rate by Zipf, among non-original valid guesses.')
    emit('"clean" = lowercase dictionary word, not a proper noun, not an inflection.')
    emit(f'{"zipf bin":10s} {"all":>6s} {"picks":>6s} {"rate%":>6s} | {"clean":>6s} {"picks":>6s} {"rate%":>6s}')

    def clean(w):
        f = feats[w]
        return f['dict_lower'] == '1' and f['proper_only'] == '0' and f['inflected'] == '0'
    for lo, hi in zip(bins, bins[1:]):
        a = [w for w in pool if lo <= z[w] < hi]
        c = [w for w in a if clean(w)]
        pa = sum(1 for w in off if lo <= z[w] < hi)
        pc = sum(1 for w in off if lo <= z[w] < hi and clean(w))
        emit(f'{lo:4.1f}-{hi:<4.1f}  {len(a):6d} {pa:6d} {100 * pa / max(len(a), 1):6.2f} | '
             f'{len(c):6d} {pc:6d} {100 * pc / max(len(c), 1):6.2f}')
    for k in ('dict_lower', 'proper_only', 'inflected'):
        emit(f'  {k:12s}: off-list picks {sum(feats[w][k] == "1" for w in off):3d}/{len(off)}'
             f'   non-orig pool {sum(feats[w][k] == "1" for w in pool):5d}/{len(pool)}')
    emit(f'  off-list picks not in lowercase dictionary: {", ".join(w for w in off if feats[w]["dict_lower"] == "0")}')
    emit()
    emit('Off-list answers (Zipf):')
    seq = [(r['date'], r['word'].upper()) for r in rows if r['word'].upper() not in orig]
    for i in range(0, len(seq), 4):
        emit('  ' + '   '.join(f'{d} {w} {z[w]:.2f}' for d, w in seq[i:i + 4]))
    emit()
    emit('Most frequent never-used clean non-original words (plausible future answers):')
    cand = sorted((w for w in pool if clean(w) and w not in used_set), key=lambda w: -z[w])[:60]
    for i in range(0, len(cand), 12):
        emit('  ' + ' '.join(cand[i:i + 12]))


def section_letters(rows):
    emit()
    emit('=' * 78)
    emit(f'3. CONSECUTIVE-DAY LETTER CORRELATION (permutation test, {N_PERM} shuffles)')
    emit('=' * 78)
    emit('ratio = observed / mean under shuffled day order (same words, random order).')
    emit('p_low = P(shuffle <= observed): small means "less shared than chance".')
    W = [r['word'].upper() for r in rows]
    N = len(W)
    P = np.array([[ord(c) - 65 for c in w] for w in W])
    A = np.zeros((N, 26), bool)
    for i in range(N):
        A[i, P[i]] = True
    rng = np.random.default_rng(0)

    def stats(idx):
        a, p = A[idx], P[idx]
        s = {}
        for lag in (1, 2, 3, 7):
            s[f'shared letters, lag {lag}'] = (a[lag:] & a[:-lag]).sum(1).mean()
        for lag in (1, 2):
            s[f'same-position matches, lag {lag}'] = (p[lag:] == p[:-lag]).sum(1).mean()
        for pos in range(5):
            s[f'same letter pos {pos + 1}, lag 1'] = (p[1:, pos] == p[:-1, pos]).mean()
        return s

    eras = [('all days', np.arange(N)),
            ('2021-2024', np.array([i for i, r in enumerate(rows) if r['date'] < '2025'])),
            ('2025+', np.array([i for i, r in enumerate(rows) if r['date'] >= '2025']))]
    for name, idx in eras:
        obs = stats(idx)
        sims = [stats(rng.permutation(idx)) for _ in range(N_PERM)]
        emit(f'-- {name} ({len(idx)} days)')
        for k, v in obs.items():
            s = np.array([x[k] for x in sims])
            emit(f'  {k:32s} obs {v:.4f}  shuffled {s.mean():.4f}  ratio {v / s.mean():.3f}'
                 f'  p_low {(np.sum(s <= v) + 1) / (N_PERM + 1):.4f}')

    emit()
    emit('Per-letter: P(letter today | letter yesterday) / P(letter today), all days')
    parts = []
    for L in range(26):
        base = A[1:, L].mean()
        prev = A[:-1, L]
        cond = A[1:, L][prev].mean() if prev.sum() else float('nan')
        parts.append(f'{chr(65 + L)} {cond / base:4.2f} (n={prev.sum()})')
    for i in range(0, 26, 6):
        emit('  ' + '   '.join(parts[i:i + 6]))


def main():
    rows, feats = load()
    section_repeats_and_offlist(rows, feats)
    section_frequency(rows, feats)
    section_letters(rows)
    path = os.path.join(ROOT, 'reports', 'answer_trends_report.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(out_lines) + '\n')
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
