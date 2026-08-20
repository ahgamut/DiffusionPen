import glob
import sys
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
)
from utils.arghandle import add_common_args
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


def main():
    parser = argparse.ArgumentParser("regen-prompts")
    parser.add_argument(
        "-n", "--num-prompts", default=1, type=int, help="number of prompts"
    )
    parser.add_argument("-o", "--output", type=str, default="./outputs/")
    parser.add_argument("--alt-text", default="./prompts/sample.txt", help="alt text")
    add_common_args(parser)

    args, m = init_generation(parser, __file__)
    IAM_TempLoader.check_preload()

    coll_xmls = list(glob.glob("./iam_data/xml/*.xml"))
    alt_words = read_words(args.alt_text)

    def distort(x):
        if "the" in x:
            return x.replace("the", "thx")
        return x

    for i in range(args.num_prompts):
        try:
            xpr = load_prompt(coll_xmls)
            while xpr is None:
                xpr = load_prompt(coll_xmls)
            raw_orig = Image.open(os.path.join("./iam_data", "forms", xpr.id + ".png"))
            raw_crop = xpr.get_cropped(raw_orig)
            s = IAM_TempLoader.map_wid_to_index(xpr.writer_id)
            max_line_width = raw_crop.width

            # same prompt, but 'the' -> 'thx'
            words = [distort(w.raw) for w in xpr.words]
            max_word_length_width = 0
            longest_word_length = max(len(word) for word in words)

            fakes, max_word_length_width = build_fake_image_N(
                words,
                s,
                args=args,
                models=m,
                longest_word_length=longest_word_length,
                max_word_length_width=max_word_length_width,
            )
            scaled_padded_words = add_rescale_padding(
                words,
                fakes,
                max_word_length_width=max_word_length_width,
                longest_word_length=longest_word_length,
            )
            regen_img = build_paragraph_image(
                scaled_padded_words, max_line_width=max_line_width
            )
            regen_img2 = build_ref_paragraph(
                fakes,
                xpr,
                max_line_width=max_line_width,
                longest_word_length=longest_word_length,
            )

            #
            words = alt_words
            max_word_length_width = 0
            longest_word_length = max(len(word) for word in words)
            fakes, max_word_length_width = build_fake_image_N(
                words,
                s,
                args=args,
                models=m,
                longest_word_length=longest_word_length,
                max_word_length_width=max_word_length_width,
            )
            scaled_padded_words = add_rescale_padding(
                words,
                fakes,
                max_word_length_width=max_word_length_width,
                longest_word_length=longest_word_length,
            )
            regen_alt = build_paragraph_image(
                scaled_padded_words, max_line_width=max_line_width
            )

            #
            rid = "%04x" % random.randint(0, 1000)
            raw_crop.save(os.path.join(args.output, f"{xpr.id}_orig.png"))
            regen_img.save(os.path.join(args.output, f"{xpr.id}_fake_{rid}.png"))
            regen_img2.save(os.path.join(args.output, f"{xpr.id}_fake-sz_{rid}.png"))
            regen_alt.save(os.path.join(args.output, f"{xpr.id}_alt_{rid}.png"))
        except Exception as e:
            print(e)
            tb = traceback.format_tb(sys.exc_info()[2])
            print("".join(tb))


if __name__ == "__main__":
    main()
