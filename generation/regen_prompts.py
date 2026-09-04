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
    build_ref_paragraph,
    build_replaced_paragraph,
    compose_on_paper,
    capture_png,
)
from utils.arghandle import add_common_args, file_check
from utils.page_prompt import (
    collect_prompt_files,
    load_page_prompt,
    WriterIndex,
    DEFAULT_ROOT,
)
from utils.gen_cli import init_generation, read_words, add_output_arg


def pick_prompt(files, dataset, data_root):
    """Load a random valid prompt for the dataset (retry on parse failures)."""
    for _ in range(len(files)):
        path = random.choice(files)
        try:
            xpr = load_page_prompt(dataset, path, data_root)
        except Exception:
            print("failed to read", path)
            print("".join(traceback.format_tb(sys.exc_info()[2])))
            xpr = None
        if xpr is not None:
            return xpr
    raise RuntimeError("no loadable prompts under " + data_root)


def make_writer_resolver(args):
    """raw writer -> style-bank row. With --writers-global, use the merged
    split's global writer ids (index the merged style bank); otherwise fall back
    to the IAM-only IAM_TempLoader (legacy IAM-339 bank), which cannot serve
    CVL/CSAFE."""
    if args.writers_global:
        widx = WriterIndex(args.writers_global)
        return lambda raw: widx.index(args.dataset, raw)
    if args.dataset == "iam":
        IAM_TempLoader.check_preload()
        return lambda raw: IAM_TempLoader.map_wid_to_index(raw)
    raise RuntimeError(
        "--dataset {} requires --writers-global (map the writer to its global id "
        "in the merged style bank)".format(args.dataset)
    )


def choose_replacements(xpr, args, mapping):
    """Pick which word positions to replace and the text to generate for each.

    ``gen`` mode: every position is eligible; the replacement text is the word's
    own transcription (a generated equivalent of the same word). ``json`` mode:
    only positions whose transcription is a key with a non-empty replacement list
    in ``mapping`` are eligible, and the replacement text is a random choice from
    ``mapping[word.raw]``. Up to K positions are chosen at random from the eligible
    set.
    """
    n = len(xpr.words)
    if args.replace_mode == "json":
        eligible = [i for i in range(n) if mapping.get(xpr.words[i].raw)]
    else:
        eligible = list(range(n))
    k = min(args.replace_k, len(eligible))
    if k < args.replace_k:
        print(f"only {len(eligible)} eligible words, replacing {k} of {args.replace_k}")
    idxs = random.sample(eligible, k) if k > 0 else []
    texts = [
        random.choice(mapping[xpr.words[i].raw])
        if args.replace_mode == "json"
        else xpr.words[i].raw
        for i in idxs
    ]
    return idxs, texts


def do_replacement(xpr, raw_orig, s, args, m, mapping):
    """Composite K generated word-swaps onto the real page and return the image
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
    return build_replaced_paragraph(
        raw_orig, xpr, gen_crops, idxs,
        blur_sigma=args.ink_blur, ink_jitter=args.ink_jitter,
    )


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
    regen_img = compose_on_paper(
        build_paragraph_image(scaled_padded_words, max_line_width=max_line_width),
        raw_orig,
        blur_sigma=args.ink_blur,
    )
    regen_img2 = build_ref_paragraph(
        fakes, xpr, raw_orig, blur_sigma=args.ink_blur, ink_jitter=args.ink_jitter
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
    regen_alt = compose_on_paper(
        build_paragraph_image(scaled_padded_words, max_line_width=max_line_width),
        raw_orig,
        blur_sigma=args.ink_blur,
    )
    return regen_img, regen_img2, regen_alt


def main():
    parser = argparse.ArgumentParser("regen-prompts")
    parser.add_argument(
        "-n", "--num-prompts", default=1, type=int, help="number of prompts"
    )
    add_output_arg(parser, default="./outputs/")
    parser.add_argument("--alt-text", default="./prompts/sample.txt", help="alt text")
    parser.add_argument(
        "--data-root", default=None,
        help="root of the raw pages+xml for --dataset (default: "
        "./iam_data, ./cvl_data, ./csafe_data)",
    )
    parser.add_argument(
        "--writers-global", default=None,
        help="writers_global.json of the merged split the style bank was built "
        "from; maps the page's writer to its global style-bank row (required for "
        "cvl/csafe)",
    )
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
        help="JSON {original_word: [replacement_words]} map, one picked at random "
        "per swap (build with utils.make_replacements); required for "
        "--replace-mode json",
    )
    parser.add_argument(
        "--capture-noise", type=float, default=5.0,
        help="sensor-noise sigma for the shared capture pass applied to every "
        "saved image (real crop AND dupes) so their file sizes/stats match; "
        "0 = re-encode only",
    )
    parser.add_argument(
        "--ink-blur", type=float, default=0.15,
        help="Gaussian sigma softening each generated crop's ink to scanner "
        "sharpness (match_ink); lower = crisper strokes, 0 = off. Saturates "
        "around 0.15 (the OpenCV kernel is ~identity below that)",
    )
    parser.add_argument(
        "--ink-jitter", type=float, default=8.0,
        help="per-word ink-darkness jitter sigma (exact-placement dupes only) so "
        "words vary in gray instead of one flat level; 0 = off",
    )
    add_common_args(parser)
    # Default dupe-realism profile (converged 2026-09-04). DOG lives in
    # add_common_args (off by default globally); the dupe pipeline turns it on.
    parser.set_defaults(dog_gs=4.0, dog_tau=25.0, dog_neg="style")

    args, m = init_generation(parser, __file__)

    # --dataset (from add_common_args) selects the page-prompt loader here.
    args.dataset = args.dataset.lower()
    if args.dataset not in DEFAULT_ROOT:
        raise RuntimeError(
            "--dataset must be one of {} for regen_prompts".format(list(DEFAULT_ROOT))
        )

    if args.replace_mode == "json" and args.replace_json is None:
        raise RuntimeError("--replace-mode json requires --replace-json")
    mapping = json.load(open(args.replace_json)) if args.replace_json else {}
    if any(not isinstance(v, list) for v in mapping.values()):
        raise RuntimeError(
            "--replace-json must map each word to a LIST of replacements "
            "(rebuild it with utils.make_replacements)"
        )

    data_root = args.data_root or DEFAULT_ROOT[args.dataset]
    resolve_writer = make_writer_resolver(args)
    files = collect_prompt_files(args.dataset, data_root)
    if not files:
        raise RuntimeError(f"no prompt files for {args.dataset} under {data_root}")
    alt_words = read_words(args.alt_text)

    for i in range(args.num_prompts):
        try:
            xpr = pick_prompt(files, args.dataset, data_root)
            raw_orig = Image.open(xpr.page_path)
            raw_crop = xpr.get_cropped(raw_orig)
            s = resolve_writer(xpr.writer_id)

            rid = "%04x" % random.randint(0, 1000)
            nz = args.capture_noise
            capture_png(raw_crop, os.path.join(args.output, f"{xpr.idd}_orig.png"), nz)

            if args.replace_k > 0:
                replaced = do_replacement(xpr, raw_orig, s, args, m, mapping)
                if replaced is not None:
                    capture_png(
                        replaced,
                        os.path.join(
                            args.output,
                            f"{xpr.idd}_replaced_{args.replace_k}_{args.replace_mode}_{rid}.png",
                        ),
                        nz,
                    )
            else:
                regen_img, regen_img2, regen_alt = regen_variants(
                    xpr, raw_orig, raw_crop, s, args, m, alt_words
                )
                capture_png(regen_img, os.path.join(args.output, f"{xpr.idd}_fake_{rid}.png"), nz)
                capture_png(regen_img2, os.path.join(args.output, f"{xpr.idd}_fake-sz_{rid}.png"), nz)
                capture_png(regen_alt, os.path.join(args.output, f"{xpr.idd}_alt_{rid}.png"), nz)
        except Exception as e:
            print(e)
            tb = traceback.format_tb(sys.exc_info()[2])
            print("".join(tb))


if __name__ == "__main__":
    main()
