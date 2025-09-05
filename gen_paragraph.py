import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, random_split
import torchvision
from tqdm import tqdm
from torch import optim
import copy
import argparse
import uuid
import json
from diffusers import AutoencoderKL, DDIMScheduler
import random
from torchvision.utils import save_image
from torch.nn import DataParallel
from torchvision import transforms
from transformers import CanineModel, CanineTokenizer

#
from models import EMA, Diffusion, UNetModel, ImageEncoder
from utils.iam_dataset import IAMDataset
from utils.GNHK_dataset import GNHK_Dataset
from utils.auxilary_functions import *
from utils.generation import (
    setup_logging,
    save_image_grid,
    crop_whitespace_width,
    build_fake_image,
    add_rescale_padding,
)
from utils.arghandle import add_common_args

OUTPUT_MAX_LEN = 95  # + 2  # <GO>+groundtruth+<END>
IMG_WIDTH = 256
IMG_HEIGHT = 64
PUNCTUATION = "_!\"#&'()*+,-./:;?"


def file_check(fname):
    if os.path.isfile(fname):
        return fname
    raise RuntimeError(f"{fname} is not a file")


def build_fakes(
    words,
    s,
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
    for word in words:
        if len(word) == longest_word_length:
            max_word_length_width = im.width
        im = build_fake_image(
            word,
            s,
            args,
            diffusion,
            ema_model,
            vae,
            feature_extractor,
            ddim,
            transform,
            tokenizer,
            text_encoder,
        )
        fakes.append(im)
    return fakes, max_word_length_width


def main():
    parser = argparse.ArgumentParser("diffusion-paragraph")
    parser.add_argument("-w", "--writer-id", type=int, default=12)
    parser.add_argument("-i", "--text-file", type=file_check, default="./sample.txt")
    parser.add_argument("-o", "--output", type=str, default="./output.png")
    parser.add_argument(
        "--max-line-width", default=900, type=int, help="max line width"
    )
    add_common_args(parser)

    args = parser.parse_args()
    print("torch version", torch.__version__)

    # create save directories
    setup_logging(args)
    torch.cuda.empty_cache()

    ############################ DATASET ############################
    transform = transforms.Compose(
        [
            # transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1, fill=255),
            transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ),  # transforms.Normalize((0.5,), (0.5,)),  #
        ]
    )

    character_classes = get_default_character_classes()

    ######################### MODEL #######################################
    vocab_size = len(character_classes)
    style_classes = 339  # for IAM Dataset
    print("Vocab size: ", vocab_size)

    if args.dataparallel == True:
        device_ids = [3, 4]
        print("using dataparallel with device:", device_ids)
    else:
        idx = int("".join(filter(str.isdigit, args.device)))
        device_ids = [idx]
    # unet = unet.to(args.device)

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
    )  # .to(args.device)

    unet = DataParallel(unet, device_ids=device_ids)
    unet = unet.to(args.device)

    # print('unet parameters')
    # print('unet', sum(p.numel() for p in unet.parameters() if p.requires_grad))

    optimizer = optim.AdamW(unet.parameters(), lr=0.0001)
    lr_scheduler = None

    mse_loss = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, args=args)

    ema = EMA(0.995)
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
        print("Loaded models and optimizer")

    if args.latent == True:
        print("VAE is true")
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
    PATH = args.style_path

    state_dict = torch.load(PATH, map_location=args.device, weights_only=True)
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

    print("Sampling started....")

    unet.load_state_dict(
        torch.load(
            f"{args.save_path}/models/ckpt.pt",
            map_location=args.device,
            weights_only=True,
        )
    )
    print("unet loaded")
    unet.eval()

    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
    ema_model.load_state_dict(
        torch.load(
            f"{args.save_path}/models/ema_ckpt.pt",
            map_location=args.device,
            weights_only=True,
        )
    )
    ema_model.eval()

    print("Sampling paragraph")
    # make the code to generate lines
    lines = open(args.text_file).read()
    words = lines.strip().split(" ")
    fakes = []
    gap = np.ones((64, 16))
    max_line_width = args.max_line_width
    total_char_count = 0
    avg_char_width = 0
    current_line_width = 0
    longest_word_length = max(len(word) for word in words)
    s = args.writer_id

    # build fake images
    fakes, max_word_length_width = build_fakes(
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
