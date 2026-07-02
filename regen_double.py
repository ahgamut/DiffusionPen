import sys
import traceback
import random
import os
import torch
import argparse

#
from utils.generation import (
    setup_logging,
    build_fake_image_N,
    add_rescale_padding,
    build_paragraph_image,
)
from utils.arghandle import add_common_args
from utils.model_setup import load_models


def file_check(fname):
    if os.path.isfile(fname):
        return fname
    raise RuntimeError(f"{fname} is not a file")


def main():
    parser = argparse.ArgumentParser("regen-double")
    parser.add_argument("-n", "--num-samples", type=int, default=5)
    parser.add_argument("-i", "--text-file", type=file_check, default="./sample.txt")
    parser.add_argument("-o", "--output", type=str, default="./outputs")
    parser.add_argument(
        "--max-line-width", default=900, type=int, help="max line width"
    )
    add_common_args(parser)

    args = parser.parse_args()
    print(__file__, "with torch", torch.__version__)

    # create save directories
    setup_logging(args)
    torch.cuda.empty_cache()

    m = load_models(args)
    diffusion = m["diffusion"]
    ema_model = m["ema_model"]
    vae = m["vae"]
    ddim = m["ddim"]
    feature_extractor = m["feature_extractor"]
    transform = m["transform"]
    tokenizer = m["tokenizer"]
    text_encoder = m["text_encoder"]

    lines = open(args.text_file).read()
    words = lines.strip().split(" ")

    for i in range(args.num_samples):
        s = random.randint(0, 338)
        try:
            # generate once
            max_line_width = args.max_line_width
            max_word_length_width = 0
            longest_word_length = max(len(word) for word in words)
            fakes, max_word_length_width = build_fake_image_N(
                words,
                s=s,
                args=args,
                diffusion=diffusion,
                ema_model=ema_model,
                vae=vae,
                feature_extractor=feature_extractor,
                ddim=ddim,
                transform=transform,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                longest_word_length=longest_word_length,
                max_word_length_width=max_word_length_width,
            )
            scaled_padded_words = add_rescale_padding(
                words,
                fakes,
                max_word_length_width=max_word_length_width,
                longest_word_length=longest_word_length,
            )
            gen_1 = build_paragraph_image(
                scaled_padded_words, max_line_width=max_line_width
            )

            # generate a second time
            max_line_width = args.max_line_width
            max_word_length_width = 0
            longest_word_length = max(len(word) for word in words)
            fakes, max_word_length_width = build_fake_image_N(
                words,
                s=s,
                args=args,
                diffusion=diffusion,
                ema_model=ema_model,
                vae=vae,
                feature_extractor=feature_extractor,
                ddim=ddim,
                transform=transform,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                longest_word_length=longest_word_length,
                max_word_length_width=max_word_length_width,
            )
            scaled_padded_words = add_rescale_padding(
                words,
                fakes,
                max_word_length_width=max_word_length_width,
                longest_word_length=longest_word_length,
            )
            gen_2 = build_paragraph_image(
                scaled_padded_words, max_line_width=max_line_width
            )

            rid = "%04x" % random.randint(0, 1000)
            gen_1.save(os.path.join(args.output, f"gen_{s}_{rid}_1.png"))
            gen_2.save(os.path.join(args.output, f"gen_{s}_{rid}_2.png"))
        except Exception as e:
            print(e)
            tb = traceback.format_tb(sys.exc_info()[2])
            print("".join(tb))


if __name__ == "__main__":
    main()
