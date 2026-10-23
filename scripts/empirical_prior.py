#!/usr/bin/env python3
"""
Empirical prior over tomorrow's Wordle answer, fitted to NYT's answer history.

Each day's answer is modelled as a two-stage choice:

  1. Category, with probabilities from exponentially weighted recent counts:
       ORIG  - original-list word never used before
       OFF   - word outside the original list, never used before
       REP   - any previously used word (repeats started 2026-02-02)
  2. Word within the category, by a conditional logit on word features:
       ORIG: zipf, zipf^2, in-dictionary, inflected, in-WordNet, offensive
       OFF:  zipf, zipf^2, in-dictionary, proper-noun, inflected, in-WordNet, offensive
       REP:  zipf, years since last use
     REP also has `recent` (last used < REP_MIN_GAP days ago).
     (in-WordNet, offensive and recent are held at 0 unless a model frees or fixes them; see `fixed`.)
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
import json
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
        self.wordnet = self.word_list('wordnet_words.txt')
        self.offensive = self.word_list('offensive_words.txt')
        zc = self.zipf - 3.0
        self.static = {
            ORIG: np.column_stack([zc, zc ** 2, self.dict_lower, self.inflected, self.wordnet, self.offensive]),
            OFF: np.column_stack([zc, zc ** 2, self.dict_lower, self.proper_only, self.inflected,
                                  self.wordnet, self.offensive]),
            REP: np.column_stack([zc]),  # plus dynamic age column
        }
        self.static_names = {
            ORIG: ['zipf', 'zipf^2', 'dict', 'inflected', 'wordnet', 'offensive'],
            OFF: ['zipf', 'zipf^2', 'dict', 'proper', 'inflected', 'wordnet', 'offensive'],
            REP: ['zipf', 'age_years', 'recent'],
        }

    def word_list(self, name):
        """0/1 indicator of membership in data/<name> (one word per line, # comments)."""
        with open(os.path.join(ROOT, 'data', name)) as f:
            s = {l.strip().upper() for l in f if l.strip() and not l.startswith('#')}
        return np.array([w in s for w in self.words], dtype=float)

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
        self.first_used = np.full(table.n, -1, dtype=np.int64)  # puzzle index of first use
        self.n_pushed = 0
        self.prev_word = None

    def pools(self):
        used = self.last_used >= 0
        return {ORIG: np.flatnonzero(self.t.in_orig & ~used),
                OFF: np.flatnonzero(~self.t.in_orig & ~used),
                REP: np.flatnonzero(used)}

    def category_of(self, word):
        if word not in self.t.index:  # answer outside valid_words.txt: count as off-list
            return OFF
        i = self.t.index[word]
        if self.last_used[i] >= 0:
            return REP
        return ORIG if self.t.in_orig[i] else OFF

    def push(self, date, word):
        self.n_pushed += 1
        if word not in self.t.index:
            self.prev_word = None
            return
        i = self.t.index[word]
        if self.first_used[i] < 0:
            self.first_used[i] = self.n_pushed - 1
        self.last_used[i] = date.toordinal()
        self.prev_word = word


# Repeat rule (see reports/model_improvement_notes.md): repeats come mostly from the first 200
# puzzles and never within ~500 days of the word's last use.
REP_EARLY = 200        # puzzle index below which a word counts as "early"
REP_EARLY_SHARE = 0.85
REP_MIN_GAP = 500      # days
REP_RECENT_MULT = 0.01


def category_sequence(table, answers):
    """Category of each answer at the time it was published."""
    hist = History(table)
    out = np.empty(len(answers), dtype=np.int64)
    for i, (date, word) in enumerate(answers):
        out[i] = hist.category_of(word)
        hist.push(date, word)
    return out


def category_prob_table(cats, half_life, pseudo):
    """(len(cats)+1, 3): row i = category probabilities from EW counts of cats[:i]."""
    decay = 0.5 ** (1.0 / half_life) if half_life else 1.0
    out = np.empty((len(cats) + 1, 3))
    w = np.full(3, float(pseudo))
    for i in range(len(cats) + 1):
        v = np.maximum(w, 1e-12)
        out[i] = v / v.sum()
        if i < len(cats):
            w *= decay
            w[cats[i]] += 1.0
    return out


def design(table, hist, cat, idx, date, lag):
    """Feature matrix (len(idx), k_cat + N_LAG) for candidates idx in category cat."""
    cols = [table.static[cat][idx]]
    if cat == REP:
        gap = date.toordinal() - hist.last_used[idx]
        cols.append(gap[:, None] / 365.25)
        cols.append((gap < REP_MIN_GAP)[:, None].astype(float))
    cols.append(lag[idx])
    return np.hstack(cols)


class DayData:
    """Per-day design rows for every answer, built once; fits on any prefix of the answers reuse them.

    For each category the rows of all days answered in that category are stacked into one matrix,
    so the likelihood is a segment softmax (np.*.reduceat) instead of a Python loop over days."""

    def __init__(self, table, answers):
        blocks = {c: ([], [], [], []) for c in range(3)}  # X blocks, chosen local row, day ordinal, answer index
        hist = History(table)
        for i, (date, word) in enumerate(answers):
            if word not in table.index:  # cannot score a word with no features
                hist.push(date, word)
                continue
            cat = hist.category_of(word)
            pool = hist.pools()[cat]
            X = design(table, hist, cat, pool, date, table.lag_features(hist.prev_word))
            b = blocks[cat]
            b[0].append(X)
            b[1].append(int(np.searchsorted(pool, table.index[word])))
            b[2].append(date.toordinal())
            b[3].append(i)
            hist.push(date, word)
        self.cats = []
        for c in range(3):
            Xs, ch, ords, ai = blocks[c]
            lens = np.array([len(x) for x in Xs], dtype=np.int64)
            starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
            X = np.vstack(Xs) if Xs else np.zeros((0, len(table.static_names[c]) + N_LAG))
            self.cats.append((X, starts, lens, starts + np.array(ch, dtype=np.int64),
                              np.array(ords, dtype=float), np.array(ai, dtype=np.int64)))

    def prefix(self, cat, n_answers):
        """Rows for days among answers[:n_answers]: (X, starts, lens, chosen rows, day ordinals)."""
        X, starts, lens, ch, ords, ai = self.cats[cat]
        k = int(np.searchsorted(ai, n_answers))
        rows = int(starts[k - 1] + lens[k - 1]) if k else 0
        return X[:rows], starts[:k], lens[:k], ch[:k], ords[:k]


def n_params(table):
    sizes = [len(table.static_names[c]) for c in range(3)]
    return sizes, sum(sizes) + N_LAG


def param_slices(table):
    sizes, total = n_params(table)
    off = np.cumsum([0] + sizes)
    lag = np.arange(sum(sizes), total)
    return [np.concatenate([np.arange(off[c], off[c + 1]), lag]) for c in range(3)], total


# Settings ./daily uses (chosen 2026-10-22 on walk-forward backtests; see
# reports/model_improvement_notes.md): no repeat within REP_MIN_GAP days (x0.01),
# offensive words x0.1, 1% floor spread over all valid words.
DAILY_SETTINGS = dict(fixed={'recent': float(np.log(0.01)), 'offensive': float(np.log(0.1))}, floor=0.01)


class PriorModel:
    """Features named in `fit` are fitted; `fixed` maps feature names to fixed coefficients.
    By default the in-WordNet, offensive and recent features are held at 0 (i.e. unused).
    rep_rule: replace the REP logit by the early-words/min-gap rule.
    floor: mix this much probability uniformly over all valid words."""

    def __init__(self, table, half_life=60.0, pseudo=0.5, l2=0.5, use_lag=True, use_features=True,
                 fit_half_life=None, fit=(), fixed=None, rep_rule=False, floor=0.0):
        self.t = table
        self.fixed = {k: v for k, v in {'wordnet': 0.0, 'offensive': 0.0, 'recent': 0.0, **(fixed or {})}.items()
                      if k not in fit}
        self.rep_rule = rep_rule
        self.floor = floor
        self.fit_half_life = fit_half_life  # days; down-weights old answers when fitting
        self.half_life = half_life
        self.pseudo = pseudo
        self.l2 = l2
        self.use_lag = use_lag
        self.use_features = use_features
        self.slices, self.k = param_slices(table)
        self.theta = np.zeros(self.k)

    # ---------- fitting ----------
    def fit(self, answers, data=None, theta0=None):
        """answers: list of (date, word) in order. `data` (a DayData built on these answers or a
        longer list starting the same way) lets repeated fits share the design rows; theta0 warm-starts."""
        if data is None:
            data = DayData(self.t, answers)
        n = len(answers)
        last = answers[-1][0].toordinal()
        blocks = []  # per category: (X, starts, lens, chosen rows, weights)
        for c in range(3):
            X, starts, lens, ch, ords = data.prefix(c, n)
            w = (0.5 ** ((last - ords) / self.fit_half_life) if self.fit_half_life else np.ones(len(ords)))
            blocks.append((c, X, starts, lens, ch, w))

        mask = np.ones(self.k, bool)
        if not self.use_features:
            mask[:] = False
            mask[self.slices[0][-N_LAG:]] = True
        if not self.use_lag:
            mask[self.slices[0][-N_LAG:]] = False
        fixed_idx, fixed_val = self.fixed_params()
        mask[fixed_idx] = False

        def evaluate(theta, derivs=True):
            ll = -0.5 * self.l2 * theta @ theta
            g = -self.l2 * theta
            H = -self.l2 * np.eye(self.k)
            for cat, X, starts, lens, ch, w in blocks:
                if len(starts) == 0:
                    continue
                sl = self.slices[cat]
                s = X @ theta[sl]
                m = np.maximum.reduceat(s, starts)
                e = np.exp(s - np.repeat(m, lens))
                z = np.add.reduceat(e, starts)
                ll += w @ (s[ch] - m - np.log(z))
                if derivs:
                    p = e / np.repeat(z, lens)
                    ex = np.add.reduceat(X * p[:, None], starts)          # (days, k_cat)
                    g[sl] += w @ (X[ch] - ex)
                    H[np.ix_(sl, sl)] -= X.T @ (X * (p * np.repeat(w, lens))[:, None]) - ex.T @ (ex * w[:, None])
            return ll, g, H

        theta = np.zeros(self.k) if theta0 is None else np.array(theta0, dtype=float)
        theta[~mask] = 0
        theta[fixed_idx] = fixed_val
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

    def fixed_params(self):
        idx, val = [], []
        for c in range(3):
            for j, name in enumerate(self.t.static_names[c]):
                if name in self.fixed:
                    idx.append(self.slices[c][j])
                    val.append(self.fixed[name])
        return np.array(idx, dtype=np.int64), np.array(val)

    # ---------- prediction ----------
    def category_probs(self, answers, upto):
        """EW counts of categories among answers[:upto]."""
        return category_prob_table(category_sequence(self.t, answers[:upto]), self.half_life, self.pseudo)[upto]

    def predict(self, hist, date, cat_probs):
        """Probability vector over all valid words for `date`, given history state."""
        t = self.t
        lag = t.lag_features(hist.prev_word)
        out = np.zeros(t.n)
        for cat, pool in hist.pools().items():
            if len(pool) == 0:
                continue
            if cat == REP and self.rep_rule:
                early = hist.first_used[pool] < REP_EARLY
                n_early = early.sum()
                share = REP_EARLY_SHARE if 0 < n_early < len(pool) else float(n_early > 0)
                p = np.where(early, share / max(n_early, 1), (1 - share) / max(len(pool) - n_early, 1))
                gap = date.toordinal() - hist.last_used[pool]
                p = p * np.where(gap < REP_MIN_GAP, REP_RECENT_MULT, 1.0)
            else:
                X = design(t, hist, cat, pool, date, lag)
                s = X @ self.theta[self.slices[cat]]
                p = np.exp(s - s.max())
            out[pool] = cat_probs[cat] * p / p.sum()
        out /= out.sum()
        if self.floor:
            out = (1 - self.floor) * out + self.floor / t.n
        return out

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
    ap.add_argument('--model-cache', metavar='JSON',
                    help='fit once per month on answers before the 1st (as in the backtest) and reuse '
                         'the coefficients from this file for the rest of the month')
    args = ap.parse_args()

    table = WordTable()
    answers = load_answers()
    model = PriorModel(table, half_life=args.half_life, **DAILY_SETTINGS)
    if args.model_cache:
        nxt = answers[-1][0] + dt.timedelta(days=1)
        month_start = nxt.replace(day=1)
        train = [a for a in answers if a[0] < month_start]
        key = {'month': str(month_start)[:7], 'n_train': len(train), 'half_life': args.half_life,
               'last_train': str(train[-1][0]), 'n_params': model.k,
               'settings': repr(DAILY_SETTINGS)}
        cached = None
        try:
            with open(args.model_cache) as f:
                cached = json.load(f)
        except (OSError, ValueError):
            pass
        if cached and cached.get('key') == key:
            model.theta = np.array(cached['theta'])
            print('Model: cached fit for', key['month'])
        else:
            model.fit(train)
            os.makedirs(os.path.dirname(os.path.abspath(args.model_cache)), exist_ok=True)
            with open(args.model_cache, 'w') as f:
                json.dump({'key': key, 'theta': model.theta.tolist()}, f)
            print('Model: refit for', key['month'])
    else:
        model.fit(answers)
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
