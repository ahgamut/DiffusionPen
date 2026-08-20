import os
import random
import torch
import numpy as np
import argparse

#
from utils.generation import (
    setup_logging,
    crop_whitespace_width,
    add_rescale_padding,
    build_paragraph_image,
    build_fake_interp_N,
    stack_images,
)
from utils.arghandle import add_common_args, file_check
from utils.model_setup import load_models


def main():
    """Main function"""
    parser = argparse.ArgumentParser("interp-viz")
    parser.add_argument("-i", "--text-file", type=file_check, default="./sample.txt")
    parser.add_argument("-o", "--output", type=str, default="./output.png")
    parser.add_argument(
        "--max-line-width", default=900, type=int, help="max line width"
    )
    add_common_args(parser)
    parser.set_defaults(interpolation=True)

    args = parser.parse_args()
    print(__file__, "with torch", torch.__version__)

    # create save directories
    setup_logging(args)
    torch.cuda.empty_cache()

    m = load_models(args)

    # make the code to generate lines
    lines = open(args.text_file).read()
    words = lines.strip().split(" ")
    max_line_width = args.max_line_width
    max_word_length_width = 0
    longest_word_length = max(len(word) for word in words)

    writers = [random.randint(0, 338) for x in range(4)]
    base_wt = 0.25
    wt_pieces = np.arange(1.00, -0.001, -base_wt)
    big_images = []

    for i in range(len(writers) - 1):
        args.writer_1 = writers[i]
        args.writer_2 = writers[i + 1]
        for wt in wt_pieces:
            if (wt == 1) and i != 0:
                continue
            args.mix_rate = wt
            # build fake images
            fakes, max_word_length_width = build_fake_interp_N(
                words,
                args=args,
                models=m,
                longest_word_length=longest_word_length,
                max_word_length_width=max_word_length_width,
            )

            # Scale and pad each word
            scaled_padded_words = add_rescale_padding(
                words,
                fakes,
                max_word_length_width=max_word_length_width,
                longest_word_length=longest_word_length,
            )

            # combine to create paragraph
            paragraph_image = build_paragraph_image(
                scaled_padded_words, max_line_width=max_line_width
            )
            big_images.append(paragraph_image)

    res_image = stack_images(big_images, margin=5)
    res_image.save(args.output)


if __name__ == "__main__":
    main()
