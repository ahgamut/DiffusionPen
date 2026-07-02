import os
import random
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision
import argparse

#
from utils.generation import (
    setup_logging,
    crop_whitespace_width,
    add_rescale_padding,
    build_paragraph_image,
)
from utils.arghandle import add_common_args, file_check
from utils.model_setup import load_models


def build_fakes_interp(
    words,
    args,
    diffusion,
    ema_model,
    vae,
    feature_extractor,
    ddim,
    transform,
    tokenizer,
    text_encoder,
    longest_word_length,
    max_word_length_width,
):
    fakes = []
    writer_1 = args.writer_1
    writer_2 = args.writer_2
    labels = torch.tensor([writer_1, writer_2]).long().to(args.device)
    ema_sampled_images = diffusion.interp_bulk(
        ema_model,
        vae,
        x_text=words,
        labels=labels,
        args=args,
        style_extractor=feature_extractor,
        noise_scheduler=ddim,
        transform=transform,
        character_classes=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        run_idx=None,
    )
    topil = torchvision.transforms.ToPILImage()
    for i in range(len(words)):
        word = words[i]
        image = ema_sampled_images[i].squeeze(0)
        im = topil(image)
        im = im.convert("L")
        im = crop_whitespace_width(im)
        im = Image.fromarray(im)
        if len(word) == longest_word_length:
            max_word_length_width = im.width
        fakes.append(im)
    return fakes, max_word_length_width


def combine_stack(images):
    res_width = max(img.width for img in images) + 10
    res_height = sum(img.height + 10 for img in images)
    dst = Image.new("RGB", (res_width, res_height), color="white")
    ch = 0
    for img in images:
        dst.paste(img, (5, ch + 5))
        ch += img.height
        ch += 5
    return dst


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
            fakes, max_word_length_width = build_fakes_interp(
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

    res_image = combine_stack(big_images)
    res_image.save(args.output)


if __name__ == "__main__":
    main()
