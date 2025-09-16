import os
import random
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision
from torch import optim
import copy
import argparse
from diffusers import AutoencoderKL, DDIMScheduler
from torch.nn import DataParallel
from torchvision import transforms
from transformers import CanineModel, CanineTokenizer

#
from models import UNetModel, ImageEncoder
from models.diffpen2 import Diffusion
from utils.auxilary_functions import *
from utils.generation import (
    setup_logging,
    crop_whitespace_width,
    add_rescale_padding,
    build_paragraph_image,
)
from utils.arghandle import add_common_args


torch.cuda.empty_cache()
OUTPUT_MAX_LEN = 95  # + 2  # <GO>+groundtruth+<END>
IMG_WIDTH = 256
IMG_HEIGHT = 64

PUNCTUATION = "_!\"#&'()*+,-./:;?"


def file_check(fname):
    if os.path.isfile(fname):
        return fname
    raise RuntimeError(f"{fname} is not a file")


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
    parser = argparse.ArgumentParser("diffusion-paragraph-interp")
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

    ############################ DATASET ############################
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    character_classes = get_default_character_classes()

    ######################### MODEL #######################################
    vocab_size = len(character_classes)
    style_classes = 339  # for IAM Dataset

    if args.dataparallel == True:
        device_ids = [3, 4]
    else:
        idx = int("".join(filter(str.isdigit, args.device)))
        device_ids = [idx]

    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    text_encoder = CanineModel.from_pretrained("google/canine-c")
    text_encoder = nn.DataParallel(text_encoder, device_ids=device_ids)
    text_encoder = text_encoder.to(args.device)

    unet = UNetModel(
        image_size=args.img_size,
        in_channels=args.channels,
        model_channels=args.emb_dim,
        out_channels=args.channels,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=(1, 1),
        channel_mult=(1, 1),
        num_heads=args.num_heads,
        num_classes=style_classes,
        context_dim=args.emb_dim,
        vocab_size=vocab_size,
        text_encoder=text_encoder,
        args=args,
    )

    unet = DataParallel(unet, device_ids=device_ids)
    unet = unet.to(args.device)

    optimizer = optim.AdamW(unet.parameters(), lr=0.0001)
    diffusion = Diffusion(img_size=args.img_size, args=args)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)

    # load from last checkpoint
    if args.load_check == True:
        unet.load_state_dict(
            torch.load(f"{args.save_path}/models/ckpt.pt", weights_only=True)
        )
        optimizer.load_state_dict(
            torch.load(f"{args.save_path}/models/optim.pt", weights_only=True)
        )
        ema_model.load_state_dict(
            torch.load(f"{args.save_path}/models/ema_ckpt.pt", weights_only=True)
        )

    if args.latent == True:
        vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
        vae = DataParallel(vae, device_ids=device_ids)
        vae = vae.to(args.device)
        # Freeze vae and text_encoder
        vae.requires_grad_(False)
    else:
        vae = None

    # add DDIM scheduler from huggingface
    ddim = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")

    #### STYLE ####
    feature_extractor = ImageEncoder(
        model_name="mobilenetv2_100", num_classes=0, pretrained=True, trainable=True
    )

    state_dict = torch.load(
        args.style_path, map_location=args.device, weights_only=True
    )
    model_dict = feature_extractor.state_dict()
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }
    model_dict.update(state_dict)
    feature_extractor.load_state_dict(model_dict)
    feature_extractor = DataParallel(feature_extractor, device_ids=device_ids)
    feature_extractor = feature_extractor.to(args.device)
    feature_extractor.requires_grad_(False)
    feature_extractor.eval()

    unet.load_state_dict(
        torch.load(
            f"{args.save_path}/models/ckpt.pt",
            map_location=args.device,
            weights_only=True,
        )
    )
    unet.eval()

    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
    ema_model.load_state_dict(
        torch.load(
            f"{args.save_path}/models/ema_ckpt.pt",
            map_location=args.device,
            weights_only=True,
        )
    )
    ema_model.eval()

    # make the code to generate lines
    lines = open(args.text_file).read()
    words = lines.strip().split(" ")
    fakes = []
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
