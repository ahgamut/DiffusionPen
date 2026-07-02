import sys
import traceback
import random
import os
import numpy as np
import torch
import argparse

#
from utils.generation import (
    setup_logging,
    build_fake_image_N,
    build_fake_interp_N,
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
    parser = argparse.ArgumentParser("regen-interp")
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
    max_line_width = args.max_line_width
    longest_word_length = max(len(word) for word in words)
    w = 0.1
    weights = np.arange(0, 1 + w, w)

    for i in range(args.num_samples):
        rid = "%04x" % random.randint(0, 1000)
        s1 = random.randint(0, 338)
        s2 = random.randint(0, 338)
        while s2 == s1:
            s2 = random.randint(0, 338)
        args.writer_1 = s1
        args.writer_2 = s2

        try:
            # generate with s1
            max_word_length_width = 0
            fakes, max_word_length_width = build_fake_image_N(
                words,
                s=s1,
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
            gen_1.save(os.path.join(args.output, f"intgen_{s1}_{rid}_1.png"))

            # generate with s2
            max_word_length_width = 0
            fakes, max_word_length_width = build_fake_image_N(
                words,
                s=s2,
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
            gen_2.save(os.path.join(args.output, f"intgen_{s2}_{rid}_1.png"))

            for weight in weights:
                # generate with interpolated style
                args.mix_rate = weight
                max_word_length_width = 0
                fakes, max_word_length_width = build_fake_interp_N(
                    words,
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
                gen_int = build_paragraph_image(
                    scaled_padded_words, max_line_width=max_line_width
                )
                gen_int.save(
                    os.path.join(
                        args.output, f"intgen_{s1}_{s2}_{weight:.1f}_{rid}.png"
                    )
                )
        except Exception as e:
            print(e)
            tb = traceback.format_tb(sys.exc_info()[2])
            print("".join(tb))


if __name__ == "__main__":
    main()
