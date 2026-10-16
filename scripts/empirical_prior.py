#!/usr/bin/env python3
"""
Empirical prior over tomorrow's Wordle answer, fitted to NYT's answer history.

Each day's answer is modelled as a two-stage choice:

  1. Category, with probabilities from exponentially weighted recent counts:
       ORIG  - original-list word never used before
       OFF   - word outside the original list, never used before
       REP   - any previously used word (repeats started 2026-02-02)
  2. Word within the category, by a conditional logit on word features:
       ORIG: zipf, zipf^2, in-dictionary, inflected
       OFF:  zipf, zipf^2, in-dictionary, proper-noun, inflected
       REP:  zipf, years since last use
     plus shared "looks like yesterday" features: same first letter,
     same last letter, number of same-position matches in positions 2-4.

Fitting is maximum likelihood (Newton's method, small L2 penalty), numpy only.

Usage:
    python3 scripts/empirical_prior.py            # fit on all data, show top words for next day
    python3 scripts/empirical_prior.py --top 50
"""

import argparse
import csv
import datetime as dt
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CATS = ('ORIG', 'OFF', 'REP')
ORIG, OFF, REP = range(3)
N_LAG = 3  # same_first, same_last, same_mid


def load_answers():
    rows = list(csv.DictReader(open(os.path.join(ROOT, 'data', 'wordle_answers.csv'))))
    return [(dt.date.fromisoformat(r['date']), r['word'].upper()) for r in rows]


class WordTable:
    """Static per-word arrays for all valid guesses."""

    def __init__(self):
        rows = list(csv.DictReader(open(os.path.join(ROOT, 'data', 'word_features.csv'))))
        self.words = [r['word'] for r in rows]
        self.index = {w: i for i, w in enumerate(self.words)}
        self.n = len(self.words)
        f = lambda k: np.array([float(r[k]) for r in rows])
        self.zipf = f('zipf')
        self.in_orig = f('in_orig').astype(bool)
        self.dict_lower = f('dict_lower')
        self.proper_only = f('proper_only')
        self.inflected = f('inflected')
        self.letters = np.array([[ord(c) - 65 for c in w] for w in self.words], dtype=np.int8)
        zc = self.zipf - 3.0
        self.static = {
            ORIG: np.column_stack([zc, zc ** 2, self.dict_lower, self.inflected]),
            OFF: np.column_stack([zc, zc ** 2, self.dict_lower, self.proper_only, self.inflected]),
            REP: np.column_stack([zc]),  # plus dynamic age column
        }
        self.static_names = {
            ORIG: ['zipf', 'zipf^2', 'dict', 'inflected'],
            OFF: ['zipf', 'zipf^2', 'dict', 'proper', 'inflected'],
            REP: ['zipf', 'age_years'],
        }

    def lag_features(self, prev_word):
        """(n, 3) array: same first letter, same last letter, # same-position in 2-4."""
        if prev_word is None:
            return np.zeros((self.n, N_LAG))
        p = self.letters[self.index[prev_word]]
        same = self.letters == p
        return np.column_stack([same[:, 0], same[:, 4], same[:, 1:4].sum(1)]).astype(float)


class History:
    """Incremental state of which words have been used and when."""

    def __init__(self, table):
        self.t = table
        self.last_used = np.full(table.n, -1, dtype=np.int64)  # day ordinal, -1 = never
        self.prev_word = None

    def pools(self):
        used = self.last_used >= 0
        return {ORIG: np.flatnonzero(self.t.in_orig & ~used),
                OFF: np.flatnonzero(~self.t.in_orig & ~used),
                REP: np.flatnonzero(used)}

    def category_of(self, word):
        i = self.t.index[word]
        if self.last_used[i] >= 0:
            return REP
        return ORIG if self.t.in_orig[i] else OFF

    def push(self, date, word):
        self.last_used[self.t.index[word]] = date.toordinal()
        self.prev_word = word


def design(table, hist, cat, idx, date, lag):
    """Feature matrix (len(idx), k_cat + N_LAG) for candidates idx in category cat."""
    cols = [table.static[cat][idx]]
    if cat == REP:
        age = (date.toordinal() - hist.last_used[idx]) / 365.25
        cols.append(age[:, None])
    cols.append(lag[idx])
    return np.hstack(cols)


def n_params(table):
    sizes = [len(table.static_names[c]) for c in range(3)]
    return sizes, sum(sizes) + N_LAG


def param_slices(table):
    sizes, total = n_params(table)
    off = np.cumsum([0] + sizes)
    lag = np.arange(sum(sizes), total)
    return [np.concatenate([np.arange(off[c], off[c + 1]), lag]) for c in range(3)], total


class PriorModel:
    def __init__(self, table, half_life=60.0, pseudo=0.5, l2=0.5, use_lag=True, use_features=True,
                 fit_half_life=None):
        self.t = table
        self.fit_half_life = fit_half_life  # days; down-weights old answers when fitting
        self.half_life = half_life
        self.pseudo = pseudo
        self.l2 = l2
        self.use_lag = use_lag
        self.use_features = use_features
        self.slices, self.k = param_slices(table)
        self.theta = np.zeros(self.k)

    # ---------- fitting ----------
    def fit(self, answers):
        """answers: list of (date, word) in order. Builds per-day design rows once, then Newton."""
        t = self.t
        hist = History(t)
        days = []  # (cat, X, chosen_row)
        for date, word in answers:
            cat = hist.category_of(word)
            pool = hist.pools()[cat]
            lag = t.lag_features(hist.prev_word)
            X = design(t, hist, cat, pool, date, lag)
            days.append((cat, X, int(np.searchsorted(pool, t.index[word])), date.toordinal()))
            hist.push(date, word)
        last = answers[-1][0].toordinal()
        days = [(c, X, ch, 0.5 ** ((last - d) / self.fit_half_life) if self.fit_half_life else 1.0)
                for c, X, ch, d in days]

        mask = np.ones(self.k, bool)
        if not self.use_features:
            mask[:] = False
            mask[self.slices[0][-N_LAG:]] = True
        if not self.use_lag:
            mask[self.slices[0][-N_LAG:]] = False
        def evaluate(theta, derivs=True):
            ll = -0.5 * self.l2 * theta @ theta
            g = -self.l2 * theta
            H = -self.l2 * np.eye(self.k)
            for cat, X, ch, w in days:
                sl = self.slices[cat]
                s = X @ theta[sl]
                m = s.max()
                e = np.exp(s - m)
                z = e.sum()
                ll += w * (s[ch] - m - np.log(z))
                if derivs:
                    p = e / z
                    ex = p @ X
                    g[sl] += w * (X[ch] - ex)
                    H[np.ix_(sl, sl)] -= w * ((X * p[:, None]).T @ X - np.outer(ex, ex))
            return ll, g, H

        theta = np.zeros(self.k)
        ll, g, H = evaluate(theta)
        for _ in range(100):
            g[~mask] = 0
            step = np.zeros(self.k)
            step[mask] = np.linalg.solve(H[np.ix_(mask, mask)], -g[mask])
            # Backtracking: the objective is concave, but full Newton steps can overshoot.
            a = 1.0
            while a > 1e-8:
                new_ll = evaluate(theta + a * step, derivs=False)[0]
                if new_ll >= ll - 1e-10:
                    break
                a *= 0.5
            theta = theta + a * step
            if np.abs(a * step).max() < 1e-6:
                break
            ll, g, H = evaluate(theta)
        self.theta = theta
        self.loglik = ll
        return self

    # ---------- prediction ----------
    def category_probs(self, answers, upto):
        """EW counts of categories among answers[:upto]."""
        hist = History(self.t)
        w = np.full(3, self.pseudo)
        decay = 0.5 ** (1.0 / self.half_life) if self.half_life else 1.0
        for date, word in answers[:upto]:
            c = hist.category_of(word)
            w *= decay
            w[c] += 1.0
            hist.push(date, word)
        w = np.maximum(w, 1e-12)
        return w / w.sum()

    def predict(self, hist, date, cat_probs):
        """Probability vector over all valid words for `date`, given history state."""
        t = self.t
        lag = t.lag_features(hist.prev_word)
        out = np.zeros(t.n)
        for cat, pool in hist.pools().items():
            if len(pool) == 0:
                continue
            X = design(t, hist, cat, pool, date, lag)
            s = X @ self.theta[self.slices[cat]]
            p = np.exp(s - s.max())
            out[pool] = cat_probs[cat] * p / p.sum()
        return out / out.sum()

    def describe(self):
        lines = []
        for c in range(3):
            names = self.t.static_names[c]
            vals = self.theta[self.slices[c][:len(names)]]
            lines.append(f'  {CATS[c]:4s}: ' + ', '.join(f'{n}={v:+.2f}' for n, v in zip(names, vals)))
        lag = self.theta[self.slices[0][-N_LAG:]]
        lines.append('  lag : ' + ', '.join(f'{n}={v:+.2f} (x{np.exp(v):.2f})'
                                          for n, v in zip(['same_first', 'same_last', 'same_mid'], lag)))
        return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--top', type=int, default=40)
    ap.add_argument('--half-life', type=float, default=60.0)
    ap.add_argument('--export', metavar='CSV', nargs='?', const=os.path.join(ROOT, 'data', 'prior_next.csv'),
                    help='write word,prob for the next day (default data/prior_next.csv), '
                         'for src/weighted_three_solver.cpp')
    args = ap.parse_args()

    table = WordTable()
    answers = load_answers()
    model = PriorModel(table, half_life=args.half_life).fit(answers)
    print('Fitted coefficients (logit scale):')
    print(model.describe())
    cp = model.category_probs(answers, len(answers))
    print('Category probabilities for next day: ' + ', '.join(f'{c}={p:.3f}' for c, p in zip(CATS, cp)))

    hist = History(table)
    for d, w in answers:
        hist.push(d, w)
    nxt = answers[-1][0] + dt.timedelta(days=1)
    p = model.predict(hist, nxt, cp)
    order = np.argsort(-p)
    print(f'\nTop {args.top} for {nxt} (after {answers[-1][1]}):')
    for r, i in enumerate(order[:args.top], 1):
        print(f'  {r:3d}. {table.words[i]}  {100 * p[i]:.3f}%  [{CATS[hist.category_of(table.words[i])]}]')
    mass = np.cumsum(p[order])
    for q in (0.5, 0.8, 0.9, 0.95, 0.99):
        print(f'  {int(q * 100)}% of mass in top {np.searchsorted(mass, q) + 1} words')
    if args.export:
        with open(args.export, 'w') as f:
            f.write('word,prob\n')
            for i in order:
                f.write(f'{table.words[i]},{p[i]:.6e}\n')
        print(f'\nwrote prior for {nxt} to {args.export}')


if __name__ == '__main__':
    main()
