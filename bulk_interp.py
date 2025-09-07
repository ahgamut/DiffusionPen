import os
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
    build_fake_interp_1,
)
from utils.arghandle import add_common_args


torch.cuda.empty_cache()
OUTPUT_MAX_LEN = 95  # + 2  # <GO>+groundtruth+<END>
IMG_WIDTH = 256
IMG_HEIGHT = 64

PUNCTUATION = "_!\"#&'()*+,-./:;?"


def img_concat(imgs):
    w = max(x.width for x in imgs)
    h = sum(x.height for x in imgs)
    dst = Image.new("RGB", (w, h))
    ch = 0
    for img in imgs:
        dst.paste(img, (0, ch))
        ch += img.height
    return dst


def main():
    """Main function"""
    parser = argparse.ArgumentParser("diffusion-bulk-interp")
    parser.add_argument("--tag", default="i1", help="tag")
    parser.add_argument("--writer-1", type=int, default=1)
    parser.add_argument("--writer-2", type=int, default=3)
    parser.add_argument("--sampling-word", type=str, default="hello")
    parser.add_argument("-o", "--output", type=str, default="./outputs/")
    add_common_args(parser)
    parser.set_defaults(interpolation=False)

    args = parser.parse_args()
    print("torch version", torch.__version__)
    args.tag = "i1"

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
    print("Vocab size: ", vocab_size)

    if args.dataparallel == True:
        device_ids = [3, 4]
        print("using dataparallel with device:", device_ids)
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

    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
    ema_model.load_state_dict(
        torch.load(
            f"{args.save_path}/models/ema_ckpt.pt",
            map_location=args.device,
            weights_only=True,
        )
    )
    ema_model.eval()

    w = 0.1
    weights = np.arange(0, 1 + w, w)
    for writer_1 in range(1, 5):
        for writer_2 in range(writer_1 + 1, 5):
            imgs = []
            for weight in weights:
                args.writer_1 = writer_1
                args.writer_2 = writer_2
                args.mix_rate = weight

                im = build_fake_interp_1(
                    args=args,
                    diffusion=diffusion,
                    ema_model=ema_model,
                    vae=vae,
                    feature_extractor=feature_extractor,
                    ddim=ddim,
                    transform=transform,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                )
                imgs.append(im)
            dst = img_concat(imgs)
            dst.save(
                os.path.join(
                    args.output,
                    f"{args.tag}-{args.sampling_word}-{writer_1}-{writer_2}.png",
                )
            )


if __name__ == "__main__":
    main()
