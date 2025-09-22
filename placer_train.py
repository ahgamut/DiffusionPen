from diffusers import AutoencoderKL, DDIMScheduler
from torch import optim
from torch.nn import DataParallel
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from transformers import CanineModel, CanineTokenizer
import argparse
import copy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

#
from models import UNetModel, ImageEncoder, EMA, Diffusion, HorizontalPlacer, AvgMeter
from utils.placer_iam import IAMPlacerDataset
from utils.auxilary_functions import *
from utils.generation import setup_logging
from utils.arghandle import add_common_args


def frz(model):
    model.eval()
    model.requires_grad_(False)


def build_IAMDataset(args, train_transform):
    full_data = IAMPlacerDataset(transforms=train_transform)
    style_classes = full_data.STYLE_CLASSES

    train_size = int(0.8 * len(full_data))
    test_size = len(full_data) - train_size
    train_data, test_data = random_split(
        full_data, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    return train_data, test_data, style_classes


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


def train_epoch(
    placer,
    diffusion,
    tokenizer,
    vae,
    optimizer,
    train_loader,
    loss_fn,
    loss_meter,
    args,
):
    placer.train()
    for i, data in enumerate(train_loader):
        wids = data[0]  # .to(args.device)
        x_cur = data[1]
        x_next = data[2]
        shifts = data[3].to(args.device)
        # (Laplace CDF?) this ensures diffs are in [-1, 1]
        # shifts = torch.sign(shifts) * (1 - torch.exp(-torch.abs(shifts)))

        x_cur["image"] = x_cur["image"].to(args.device)
        x_cur["text_features"] = tokenizer(
            x_cur["text"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=40,
        ).to(args.device)
        #
        x_next["image"] = x_next["image"].to(args.device)
        x_next["text_features"] = tokenizer(
            x_next["text"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=40,
        ).to(args.device)

        if np.random.random() < 0.1:
            # try with reconstructions?
            pass

        predicted_shifts = placer(x_cur, x_next)
        loss = loss_fn(shifts, predicted_shifts)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        count = x_cur["image"].size(0)
        loss_meter.update(loss.item(), count)
    print("train", repr(loss_meter))


def val_epoch(
    placer, diffusion, tokenizer, vae, test_loader, loss_fn, loss_meter, args
):
    placer.eval()
    for i, data in enumerate(test_loader):
        wids = data[0]  # .to(args.device)
        x_cur = data[1]
        x_next = data[2]
        shifts = data[3].to(args.device)
        # (Laplace CDF?) this ensures diffs are in [-1, 1]
        # shifts = torch.sign(shifts) * (1 - torch.exp(-torch.abs(shifts)))

        x_cur["image"] = x_cur["image"].to(args.device)
        x_cur["text_features"] = tokenizer(
            x_cur["text"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=40,
        ).to(args.device)
        #
        x_next["image"] = x_next["image"].to(args.device)
        x_next["text_features"] = tokenizer(
            x_next["text"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=40,
        ).to(args.device)

        if np.random.random() < 0.1:
            # try with reconstructions?
            pass

        predicted_shifts = placer(x_cur, x_next)
        loss = loss_fn(shifts, predicted_shifts)
        err = torch.abs(shifts - predicted_shifts)
        print(
            torch.sum(err > 0.02).item(),
            "pixels off by >0.02,",
            torch.sum(err <= 0.02).item(),
            "pixels off by <=0.02",
        )
        # print(shifts - predicted_shifts)

        count = x_cur["image"].size(0)
        loss_meter.update(loss.item(), count)
    print("validation", repr(loss_meter))


def train(
    placer,
    diffusion,
    model,
    ema,
    ema_model,
    vae,
    optimizer,
    loss_fn,
    train_loader,
    test_loader,
    num_classes,
    style_extractor,
    vocab_size,
    noise_scheduler,
    transforms,
    args,
    tokenizer=None,
    text_encoder=None,
    lr_scheduler=None,
):
    model.train()
    loss_meter = AvgMeter("MSE")
    print("Training started....")

    for epoch in range(args.epochs):
        print("Epoch:", epoch)
        train_epoch(
            placer=placer,
            diffusion=diffusion,
            tokenizer=tokenizer,
            vae=vae,
            optimizer=optimizer,
            train_loader=train_loader,
            loss_fn=loss_fn,
            loss_meter=loss_meter,
            args=args,
        )

        if epoch % 10 == 0:
            val_epoch(
                placer=placer,
                diffusion=diffusion,
                tokenizer=tokenizer,
                vae=vae,
                test_loader=test_loader,
                loss_fn=loss_fn,
                loss_meter=loss_meter,
                args=args,
            )
            torch.save(
                placer.state_dict(),
                os.path.join(args.save_path, "models", "placer_ckpt.pt"),
            )
            torch.save(
                optimizer.state_dict(),
                os.path.join(args.save_path, "models", "placer_optim.pt"),
            )


def custom_loss(out=1.0, alpha=0.5, beta=2.0):
    def fn(pred, target):
        l2 = nn.functional.mse_loss(pred, target, reduction="none")
        small = alpha * l2[l2 <= out].sum()
        big = beta * l2[l2 > out].sum()
        return big + small

    return fn


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
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ),  # transforms.Normalize((0.5,), (0.5,)),  #
        ]
    )
    #
    if args.dataset == "iam":
        print("loading IAM")
        train_data, test_data, style_classes = build_IAMDataset(args, train_transform)

    else:
        raise ValueError("unknown dataset!")

    print(len(train_data))
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

    loss_fn = custom_loss(0.04, alpha=1.0, beta=5.0)
    diffusion = Diffusion(img_size=args.img_size, args=args)

    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet)

    # load from last checkpoint
    if args.load_check == True:
        unet.load_state_dict(
            torch.load(f"{args.save_path}/models/ckpt.pt", weights_only=True)
        )
        ema_model.load_state_dict(
            torch.load(f"{args.save_path}/models/ema_ckpt.pt", weights_only=True)
        )

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
    placer = DataParallel(placer, device_ids=device_ids)
    optimizer = optim.AdamW(placer.parameters(), lr=0.001)
    placer_wts_path = f"{args.save_path}/models/placer_ckpt.pt"
    if os.path.isfile(placer_wts_path):
        placer.load_state_dict(torch.load(placer_wts_path, weights_only=True))

    placer_optim_path = f"{args.save_path}/models/placer_optim.pt"
    if os.path.isfile(placer_optim_path):
        optimizer.load_state_dict(torch.load(placer_optim_path, weights_only=True))
    placer = placer.to(args.device)

    ## freeze everyone except the placer model
    frz(unet)
    frz(vae)
    frz(text_encoder)
    frz(feature_extractor)
    frz(ema_model)

    #
    train(
        placer=placer,
        diffusion=diffusion,
        model=unet,
        ema=ema,
        ema_model=ema_model,
        vae=vae,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        test_loader=test_loader,
        num_classes=style_classes,
        style_extractor=feature_extractor,
        vocab_size=vocab_size,
        noise_scheduler=ddim,
        transforms=train_transform,
        args=args,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        lr_scheduler=None,
    )


if __name__ == "__main__":
    main()
