#!/usr/bin/env python3
"""
Build per-word features for every valid Wordle guess, used by the empirical
answer-prior model.

Output: data/word_features.csv with columns
    word         uppercase 5-letter word
    zipf         wordfreq Zipf frequency (log10 per-billion-words + 9; 0 = unseen)
    in_orig      1 if in the original curated answer list (data/wordle_words.txt)
    dict_lower   1 if a lowercase entry exists in /usr/share/dict/american-english
    proper_only  1 if the dictionary only has a capitalised form (proper noun)
    inflected    1 if the word looks like a plural (-S/-ES) or past tense (-ED)
                 of a dictionary word

Requires: pip install wordfreq  (only this script needs it)
"""

import os
import sys

from wordfreq import zipf_frequency

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DICT_PATH = '/usr/share/dict/american-english'


def load_words(path):
    with open(path) as f:
        return [w.strip().upper() for w in f if w.strip()]


def main():
    valid = load_words(os.path.join(ROOT, 'valid_words.txt'))
    orig = set(load_words(os.path.join(ROOT, 'data', 'wordle_words.txt')))

    lower, capital = set(), set()
    if os.path.exists(DICT_PATH):
        with open(DICT_PATH, encoding='utf-8', errors='ignore') as f:
            for line in f:
                w = line.strip()
                if not w.isalpha() or not w.isascii():
                    continue
                (lower if w.islower() else capital).add(w.lower())
    else:
        print(f'warning: {DICT_PATH} missing; dictionary features will be 0', file=sys.stderr)

    def inflected(l):
        if l.endswith('s') and not l.endswith(('ss', 'us', 'is')):
            if l[:-1] in lower or (l.endswith('es') and l[:-2] in lower) \
                    or (l.endswith('ies') and l[:-3] + 'y' in lower):
                return 1
        if l.endswith('ed'):
            if l[:-2] in lower or l[:-1] in lower \
                    or (l.endswith('ied') and l[:-3] + 'y' in lower):
                return 1
        return 0

    out = os.path.join(ROOT, 'data', 'word_features.csv')
    with open(out, 'w') as f:
        f.write('word,zipf,in_orig,dict_lower,proper_only,inflected\n')
        for w in valid:
            l = w.lower()
            f.write(f'{w},{zipf_frequency(l, "en"):.2f},{int(w in orig)},'
                    f'{int(l in lower)},{int(l in capital and l not in lower)},'
                    f'{inflected(l)}\n')
    print(f'wrote {len(valid)} rows to {out}')


if __name__ == '__main__':
    main()
