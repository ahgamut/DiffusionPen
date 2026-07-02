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
from utils.arghandle import add_common_args, file_check
from utils.model_setup import load_models


def main():
    parser = argparse.ArgumentParser("gen-paragraph")
    parser.add_argument("-w", "--writer-id", type=int, default=12)
    parser.add_argument("-i", "--text-file", type=file_check, default="./sample.txt")
    parser.add_argument("-o", "--output", type=str, default="./output.png")
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

    # make the code to generate lines
    lines = open(args.text_file).read()
    words = lines.strip().split(" ")
    max_line_width = args.max_line_width
    max_word_length_width = 0
    longest_word_length = max(len(word) for word in words)
    s = args.writer_id

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
    paragraph_image.save(args.output)


if __name__ == "__main__":
    main()
