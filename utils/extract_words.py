"""Dump the unique words from a folder of IAM form-XML files, one per line.

Point ``--input`` at a folder holding IAM ``<form>.xml`` files (e.g. the raw
``iam_data/xml/`` dir) and get a text file of every distinct word. Each ``<word>``
element's ``text`` is unescaped exactly as ``utils.subprompt.Word.from_elem``
does it, so an emitted token is exactly a ``word.raw`` -- a valid key for
``--replace-mode json`` in ``regen_prompts.py`` (fill ``replacements.json`` from
this list). Stdlib only (no PIL/torch), so it runs anywhere.

    python -m utils.extract_words iam_data/xml -o words.txt
"""

import argparse
import glob
import os
import sys
import traceback
import xml.etree.ElementTree as ET
from xml.sax.saxutils import unescape as _unescape


def unescape(x):
    # mirror utils.subprompt.unescape so keys match regen_prompts' word.raw
    return _unescape(x, {"&quot;": '"', "&apos;": "'"})


def collect_words(files, alpha_only=False):
    """Union of every ``<word>``'s unescaped text over the parseable XMLs."""
    words = set()
    for path in files:
        try:
            root = ET.parse(path).getroot()
        except Exception:
            print("failed to parse", path, file=sys.stderr)
            print("".join(traceback.format_tb(sys.exc_info()[2])), file=sys.stderr)
            continue
        for w in root.iter("word"):
            text = w.attrib.get("text")
            if text is None:
                continue
            raw = unescape(text)
            if alpha_only and not any(c.isalpha() for c in raw):
                continue
            words.add(raw)
    return words


def main():
    parser = argparse.ArgumentParser("extract-words")
    parser.add_argument("input", help="folder containing IAM form-XML files")
    parser.add_argument("-o", "--output", default="words.txt", help="output text file")
    parser.add_argument(
        "--alpha-only", action="store_true",
        help="drop tokens with no letter (pure punctuation/numbers)",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input, "**", "*.xml"), recursive=True))
    if not files:
        raise RuntimeError("no .xml files under " + args.input)

    words = collect_words(files, alpha_only=args.alpha_only)
    with open(args.output, "w") as f:
        for word in sorted(words):
            f.write(word + "\n")
    print(f"{len(words)} unique words from {len(files)} files -> {args.output}")


if __name__ == "__main__":
    main()
