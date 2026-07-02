import os
import sys
import traceback
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


def range_check(x):
    l, u = x.split("-")
    l = int(l)
    u = int(u)

    if l < 0 or u < 0 or l > u:
        raise RuntimeError(f"invalid range: {x}")

    return (l, u)


def main():
    """Main function"""
    parser = argparse.ArgumentParser("diffusion-paragraph-bulk")
    parser.add_argument("-w", "--writer-range", type=range_check, default=(1, 1))
    parser.add_argument(
        "--max-line-width", default=900, type=int, help="max line width"
    )
    parser.add_argument("-i", "--text-file", type=file_check, default="./sample.txt")
    parser.add_argument("-o", "--output", type=str, default="./output.png")
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

    # make the code to generate lines
    lines = open(args.text_file).read()
    words = lines.strip().split(" ")
    max_line_width = args.max_line_width
    max_word_length_width = 0
    longest_word_length = max(len(word) for word in words)

    output_template = args.output.replace(".png", "-{s}.png")
    writer_range = args.writer_range
    for s in range(writer_range[0], writer_range[1] + 1):
        try:
            # build fake images
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
            paragraph_image.save(output_template.format(s=s))
        except Exception as e:
            print("failed for", s)
            print(e)
            tb = traceback.format_tb(sys.exc_info()[2])
            print("".join(tb))


if __name__ == "__main__":
    main()
