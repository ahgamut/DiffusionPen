import glob
import sys
import json
import traceback
import random
import os
import argparse
from PIL import Image

#
from utils.iam_temploader import IAM_TempLoader
from utils.generation import (
    build_fake_image_N,
    add_rescale_padding,
    build_paragraph_image,
    build_replaced_paragraph,
)
from utils.arghandle import add_common_args, file_check
from utils.subprompt import Prompt as XMLPrompt
from utils.gen_cli import init_generation, read_words


def build_ref_paragraph(fakes, xpr, max_line_width, longest_word_length):
    assert len(xpr.words) == len(fakes)
    dupe = Image.new("RGB", size=(xpr.img_width, xpr.img_height), color="white")

    for i in range(len(fakes)):
        word = xpr.words[i]
        fake = fakes[i]
        ratio = word.height / fake.height
        #
        scaled_width = int(fake.width * ratio)
        scaled_height = word.height
        scaled_img = fakes[i].resize((scaled_width, scaled_height), Image.LANCZOS)
        dupe.paste(scaled_img, (word.x_start, word.y_start))

    dupe = dupe.convert("L")
    return xpr.get_cropped(dupe)


def load_prompt(coll):
    xpr = None
    fname = None
    try:
        fname = random.choice(coll)
        xpr = XMLPrompt(fname)
        assert xpr.writer_id in IAM_TempLoader.wr_dict
    except Exception:
        print(f"failed to read {fname}")
        tb = traceback.format_tb(sys.exc_info()[2])
        print("".join(tb))
        xpr = None
    return xpr


def choose_replacements(xpr, args, mapping):
    """Pick which word positions to replace and the text to generate for each.

    ``gen`` mode: every position is eligible; the replacement text is the word's
    own transcription (a generated equivalent of the same word). ``json`` mode:
    only positions whose transcription is a key in ``mapping`` are eligible, and
    the replacement text is ``mapping[word.raw]`` (a different word). Up to K
    positions are chosen at random from the eligible set.
    """
    n = len(xpr.words)
    if args.replace_mode == "json":
        eligible = [i for i in range(n) if xpr.words[i].raw in mapping]
    else:
        eligible = list(range(n))
    k = min(args.replace_k, len(eligible))
    if k < args.replace_k:
        print(f"only {len(eligible)} eligible words, replacing {k} of {args.replace_k}")
    idxs = random.sample(eligible, k) if k > 0 else []
    texts = [
        mapping[xpr.words[i].raw] if args.replace_mode == "json" else xpr.words[i].raw
        for i in idxs
    ]
    return idxs, texts


def do_replacement(xpr, raw_orig, s, args, m, mapping):
    """Composite K generated word-swaps onto the real form and return the image
    (or None when there is nothing eligible to replace)."""
    idxs, texts = choose_replacements(xpr, args, mapping)
    if not idxs:
        print("nothing to replace for", xpr.idd)
        return None
    longest_word_length = max(len(t) for t in texts)
    gen_crops, _ = build_fake_image_N(
        texts,
        s,
        args=args,
        models=m,
        longest_word_length=longest_word_length,
        max_word_length_width=0,
    )
    return build_replaced_paragraph(raw_orig, xpr, gen_crops, idxs)


def regen_variants(xpr, raw_orig, raw_crop, s, args, m, alt_words):
    """Original behaviour: full-paragraph regeneration in three variants
    (heuristic reflow, exact-XML placement, and alt-text reflow)."""
    max_line_width = raw_crop.width

    # same prompt, regenerated
    words = [w.raw for w in xpr.words]
    longest_word_length = max(len(word) for word in words)
    fakes, max_word_length_width = build_fake_image_N(
        words, s, args=args, models=m,
        longest_word_length=longest_word_length, max_word_length_width=0,
    )
    scaled_padded_words = add_rescale_padding(
        words, fakes,
        max_word_length_width=max_word_length_width,
        longest_word_length=longest_word_length,
    )
    regen_img = build_paragraph_image(scaled_padded_words, max_line_width=max_line_width)
    regen_img2 = build_ref_paragraph(
        fakes, xpr, max_line_width=max_line_width,
        longest_word_length=longest_word_length,
    )

    # alt text, reflowed
    words = alt_words
    longest_word_length = max(len(word) for word in words)
    fakes, max_word_length_width = build_fake_image_N(
        words, s, args=args, models=m,
        longest_word_length=longest_word_length, max_word_length_width=0,
    )
    scaled_padded_words = add_rescale_padding(
        words, fakes,
        max_word_length_width=max_word_length_width,
        longest_word_length=longest_word_length,
    )
    regen_alt = build_paragraph_image(scaled_padded_words, max_line_width=max_line_width)
    return regen_img, regen_img2, regen_alt


def main():
    parser = argparse.ArgumentParser("regen-prompts")
    parser.add_argument(
        "-n", "--num-prompts", default=1, type=int, help="number of prompts"
    )
    parser.add_argument("-o", "--output", type=str, default="./outputs/")
    parser.add_argument("--alt-text", default="./prompts/sample.txt", help="alt text")
    parser.add_argument(
        "--replace-k", type=int, default=0,
        help="randomly replace K real words with generated crops in place "
        "(0 = disabled, run the original full-regen variants instead)",
    )
    parser.add_argument(
        "--replace-mode", choices=["gen", "json"], default="gen",
        help="gen: regenerate the same word text; json: swap to a different word "
        "from --replace-json",
    )
    parser.add_argument(
        "--replace-json", type=file_check, default=None,
        help="JSON {original_word: replacement_word} map (required for "
        "--replace-mode json)",
    )
    add_common_args(parser)

    args, m = init_generation(parser, __file__)
    IAM_TempLoader.check_preload()

    if args.replace_mode == "json" and args.replace_json is None:
        raise RuntimeError("--replace-mode json requires --replace-json")
    mapping = json.load(open(args.replace_json)) if args.replace_json else {}

    coll_xmls = list(glob.glob("./iam_data/xml/*.xml"))
    alt_words = read_words(args.alt_text)

    for i in range(args.num_prompts):
        try:
            xpr = load_prompt(coll_xmls)
            while xpr is None:
                xpr = load_prompt(coll_xmls)
            raw_orig = Image.open(os.path.join("./iam_data", "forms", xpr.idd + ".png"))
            raw_crop = xpr.get_cropped(raw_orig)
            s = IAM_TempLoader.map_wid_to_index(xpr.writer_id)

            rid = "%04x" % random.randint(0, 1000)
            raw_crop.save(os.path.join(args.output, f"{xpr.idd}_orig.png"))

            if args.replace_k > 0:
                replaced = do_replacement(xpr, raw_orig, s, args, m, mapping)
                if replaced is not None:
                    replaced.save(
                        os.path.join(
                            args.output,
                            f"{xpr.idd}_replaced_{args.replace_k}_{args.replace_mode}_{rid}.png",
                        )
                    )
            else:
                regen_img, regen_img2, regen_alt = regen_variants(
                    xpr, raw_orig, raw_crop, s, args, m, alt_words
                )
                regen_img.save(os.path.join(args.output, f"{xpr.idd}_fake_{rid}.png"))
                regen_img2.save(os.path.join(args.output, f"{xpr.idd}_fake-sz_{rid}.png"))
                regen_alt.save(os.path.join(args.output, f"{xpr.idd}_alt_{rid}.png"))
        except Exception as e:
            print(e)
            tb = traceback.format_tb(sys.exc_info()[2])
            print("".join(tb))


if __name__ == "__main__":
    main()
