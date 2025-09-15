from diffusers import AutoencoderKL, DDIMScheduler
from torch import optim
from torch.nn.functional import cross_entropy
from torch.nn import DataParallel
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from transformers import CanineModel, CanineTokenizer
import argparse
import copy
import json
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

#
from models import UNetModel, ImageEncoder, EMA, Diffusion, HorizontalPlacer
from utils.cvl_dataset import CVLDataset
from utils.iam_dataset import IAMDataset
from utils.GNHK_dataset import GNHK_Dataset
from utils.auxilary_functions import *
from utils.generation import save_image_grid, setup_logging
from utils.arghandle import add_common_args


def frz(model):
    model.eval()
    model.requires_grad_(False)


def build_IAMDataset(args):
    pass


def load_style_weights(model, device, style_path):
    print(
        "Number of model parameters: {}".format(
            sum([p.data.nelement() for p in model.parameters()])
        )
    )
    state_dict = torch.load(style_path, map_location=device, weights_only=True)
    model_dict = model.state_dict()
    sub_dict = dict()
    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            sub_dict[k] = v
        else:
            print("skipping style weights for: ", k)
    model_dict.update(sub_dict)
    model.load_state_dict(model_dict)
    print("Pretrained style model loaded")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=320)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--style-name", default="mobilenetv2_100", type=str)
    add_common_args(parser)
    args = parser.parse_args()
    setup_logging(args)
    #
    if args.dataset == "iam":
        print("loading IAM")
        train_data, test_data, style_classes = build_IAMDataset(args)

    else:
        raise ValueError("unknown dataset!")

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    character_classes = get_default_character_classes()

    ###
    vocab_size = len(character_classes)
    print("Vocab size: ", vocab_size)

    if args.dataparallel == True:
        device_ids = [3, 4]
        print("using dataparallel with device:", device_ids)
    else:
        idx = int("".join(filter(str.isdigit, args.device)))
        device_ids = [idx]

    ####
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
    lr_scheduler = None

    mse_loss = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, args=args)

    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet)

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
    else:
        vae = None

    # add DDIM scheduler from huggingface
    ddim = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")

    #### STYLE ####
    if args.style_name == "mobilenetv2_100":
        feature_extractor = ImageEncoder(
            model_name="mobilenetv2_100", num_classes=0, pretrained=True, trainable=True
        )
    elif args.style_name == "resnet18":
        feature_extractor = ImageEncoder(
            model_name="resnet18", num_classes=0, pretrained=True, trainable=True
        )
    else:
        raise ValueError(f"unable to load style model {style_name}!")
    load_style_weights(feature_extractor, args.device, args.style_path)
    feature_extractor = DataParallel(feature_extractor, device_ids=device_ids)
    feature_extractor = feature_extractor.to(args.device)

    ### PLACER
    placer = HorizontalPlacer(
        text_encoder=text_encoder, style_encoder=feature_extractor
    )
    placer_wts_path = f"{args.save_path}/models/placer_ckpt.pt"
    if os.path.isfile(placer_wts_path):
        placer.load_state_dict(torch.load(placer_wts_path, weights_only=True))

    ## freeze everyone except the placer model
    frz(tokenizer)
    frz(unet)
    frz(vae)
    frz(text_encoder)
    frz(feature_extractor)
    frz(ema_model)


if __name__ == "__main__":
    main()
