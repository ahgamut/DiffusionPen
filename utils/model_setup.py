import copy
import torch
import torch.nn as nn
import torchvision
from torch.nn import DataParallel
from torchvision import transforms
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CanineModel, CanineTokenizer

import os

from models import UNetModel, ImageEncoder, Diffusion, WordPlacer, WordUpsampler
from utils.auxiliary_functions import get_default_character_classes


def load_models(args):
    character_classes = get_default_character_classes()
    vocab_size = len(character_classes)
    style_classes = 339  # IAM Dataset

    idx = int("".join(filter(str.isdigit, args.device)))
    device_ids = [idx, idx + 1] if args.dataparallel else [idx]

    transform = transforms.Compose([
        transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

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

    diffusion = Diffusion(img_size=args.img_size, args=args)

    # Optional precomputed per-writer style bank (stage-3 Part A). When enabled
    # and present, the sampling path looks up a writer's mean style vector
    # instead of reading 5 random crops + running the CNN per word.
    if getattr(args, "style_bank", False):
        bank_path = getattr(args, "style_bank_path", None)
        if bank_path and os.path.isfile(bank_path):
            bank = torch.load(bank_path, map_location=args.device, weights_only=True)
            diffusion.style_bank = bank.to(args.device)
            print("loaded style bank from", bank_path, tuple(bank.shape))

    if args.latent:
        vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
        vae = DataParallel(vae, device_ids=device_ids)
        vae = vae.to(args.device)
        vae.requires_grad_(False)
    else:
        vae = None

    ddim = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")

    feature_extractor = ImageEncoder(
        model_name="mobilenetv2_100", num_classes=0, pretrained=True, trainable=True
    )
    style_state_dict = torch.load(
        args.style_path, map_location=args.device, weights_only=True
    )
    model_dict = feature_extractor.state_dict()
    style_state_dict = {
        k: v
        for k, v in style_state_dict.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }
    model_dict.update(style_state_dict)
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

    # Optional autoregressive word placer (learned placement). Absent -> None,
    # and callers fall back to the heuristic layout with no regression.
    placer = None
    placer_path = getattr(args, "placer_path", None)
    if placer_path and os.path.isfile(placer_path):
        placer = WordPlacer(num_writers=style_classes)
        placer = DataParallel(placer, device_ids=device_ids)
        placer.load_state_dict(
            torch.load(placer_path, map_location=args.device, weights_only=True)
        )
        placer = placer.to(args.device)
        placer.eval()
        placer.requires_grad_(False)
        print("loaded word placer from", placer_path)

    # Optional learned super-resolution upsampler. Absent -> None, and callers
    # fall back to Lanczos.
    upsampler = None
    upsampler_path = getattr(args, "upsampler_path", None)
    if upsampler_path and os.path.isfile(upsampler_path):
        upsampler = WordUpsampler()
        upsampler = DataParallel(upsampler, device_ids=device_ids)
        upsampler.load_state_dict(
            torch.load(upsampler_path, map_location=args.device, weights_only=True)
        )
        upsampler = upsampler.to(args.device)
        upsampler.eval()
        upsampler.requires_grad_(False)
        print("loaded word upsampler from", upsampler_path)

    return {
        "transform": transform,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "diffusion": diffusion,
        "ema_model": ema_model,
        "vae": vae,
        "ddim": ddim,
        "feature_extractor": feature_extractor,
        "placer": placer,
        "upsampler": upsampler,
    }
