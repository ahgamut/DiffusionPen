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
from utils.relcharsize import build_placed_paragraph
from utils.arghandle import add_common_args, file_check
from utils.model_setup import load_models


def main():
    parser = argparse.ArgumentParser("gen-rcs")
    parser.add_argument("-w", "--writer-id", type=int, default=12)
    parser.add_argument("-i", "--text-file", type=file_check, default="./sample.txt")
    parser.add_argument("-o", "--output", type=str, default="./output.png")
    parser.add_argument(
        "--max-line-width", default=900, type=int, help="max line width"
    )
    parser.add_argument("--font-size", default=16, type=int, help="font size")
    parser.add_argument("--dpi", default=300, help="DPI")
    parser.add_argument(
        "--image-aspect",
        action="store_true",
        dest="use_aspect",
        help="use aspect ratio from image",
    )
    parser.add_argument(
        "--font-aspect",
        action="store_false",
        dest="use_aspect",
        help="use aspect ratio from font",
    )

    add_common_args(parser)
    parser.set_defaults(use_aspect=True)

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
        crop_whitespace=True,
    )

    # combine to create paragraph
    paragraph_image = build_placed_paragraph(
        words,
        fakes,
        max_line_width=max_line_width,
        font_size=args.font_size,
        dpi=args.dpi,
        use_aspect=args.use_aspect,
    )
    paragraph_image.save(args.output)


if __name__ == "__main__":
    main()
