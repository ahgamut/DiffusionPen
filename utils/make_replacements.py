"""Build a replacements.json for regen_prompts.py from a word list, via wordfreq.

Given a text file with one word per line (e.g. the output of
``utils.extract_words``), emit a JSON map ``{word: [k replacements]}`` where each
replacement is a real, similarly-common word of roughly the same length (within
``--length-tol`` letters) and the same capitalization pattern as the source word.
The k candidates are a random sample of the length-matched pool, so re-runs vary
(pass ``--seed`` to fix them); ``regen_prompts.py --replace-mode json`` then picks
one at random per substitution.

Candidates come from wordfreq's frequency-ordered English list (lowercased), so
this does no name/POS reasoning -- a capitalized token just gets same-length
common words, Title-cased. Only source words whose length is within
``--min-length``..``--max-length`` are considered; tokens with no letter (pure
punctuation/numbers) and tokens with no length match are skipped (omitted).

    python -m utils.make_replacements words.txt -o replacements.json -k 3
"""

import argparse
import json
import random

from wordfreq import top_n_list


def build_length_buckets(lang, pool_size):
    """Frequency-ordered, alphabetic-only words bucketed by length: {len: [words]}."""
    buckets = {}
    for w in top_n_list(lang, pool_size):
        if not w.isalpha():
            continue
        buckets.setdefault(len(w), []).append(w)
    return buckets


def apply_case(cand, word):
    """Re-apply ``word``'s capitalization pattern to a lowercase candidate."""
    if word.isupper():
        return cand.upper()
    if word[:1].isupper():
        return cand[:1].upper() + cand[1:]
    return cand


def candidates_for(word, buckets, tol):
    """Lowercase length-matched candidates (excluding the word itself)."""
    lo = word.lower()
    n = len(word)
    pool = []
    for length in range(n - tol, n + tol + 1):
        for c in buckets.get(length, ()):
            if c != lo:
                pool.append(c)
    return pool


def build_replacements(words, buckets, k, tol, rng, min_len=3, max_len=8):
    out = {}
    for word in words:
        if not (min_len <= len(word) <= max_len):
            continue  # only replace words in the requested length range
        if not any(ch.isalpha() for ch in word):
            continue  # pure punctuation / numbers -- nothing to replace
        pool = candidates_for(word, buckets, tol)
        if not pool:
            continue
        picks = rng.sample(pool, min(k, len(pool)))
        out[word] = [apply_case(c, word) for c in picks]
    return out


def read_words(path):
    with open(path) as f:
        seen, words = set(), []
        for line in f:
            w = line.strip()
            if w and w not in seen:
                seen.add(w)
                words.append(w)
    return words


def main():
    parser = argparse.ArgumentParser("make-replacements")
    parser.add_argument("input", help="text file with one word per line")
    parser.add_argument("-o", "--output", default="replacements.json")
    parser.add_argument(
        "-k", "--per-word", type=int, default=3, help="replacements per word"
    )
    parser.add_argument("--lang", default="en", help="wordfreq language code")
    parser.add_argument(
        "--length-tol", type=int, default=1,
        help="allowed letter-count difference from the source word",
    )
    parser.add_argument(
        "--min-length", type=int, default=3,
        help="only replace source words at least this long",
    )
    parser.add_argument(
        "--max-length", type=int, default=8,
        help="only replace source words at most this long",
    )
    parser.add_argument(
        "--pool-size", type=int, default=50000,
        help="how many of the most frequent words to draw candidates from "
        "(raise it for better coverage of long/rare words)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="RNG seed for reproducible samples"
    )
    args = parser.parse_args()

    words = read_words(args.input)
    buckets = build_length_buckets(args.lang, args.pool_size)
    rng = random.Random(args.seed)
    out = build_replacements(
        words, buckets, args.per_word, args.length_tol, rng,
        min_len=args.min_length, max_len=args.max_length,
    )

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    print(f"{len(out)}/{len(words)} words got replacements -> {args.output}")


if __name__ == "__main__":
    main()
