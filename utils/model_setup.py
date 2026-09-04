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

# CANINE is cached after the first run; skip the per-run hub staleness check.
# Set DIFFPEN_HF_ONLINE=1 for the one-time fetch on a cold cache.
_CANINE_LOCAL_ONLY = os.environ.get("DIFFPEN_HF_ONLINE", "0") != "1"


def _embedding_rows(state_dict, suffix):
    """Row count of the first param whose key ends with ``suffix`` (an
    ``nn.Embedding.weight``), or None. Used to size a model to match its own
    checkpoint's writer-class count -- IAM 339 or the merged W, transparently."""
    for k, v in state_dict.items():
        if k.endswith(suffix):
            return int(v.shape[0])
    return None


def load_models(args):
    character_classes = get_default_character_classes()
    vocab_size = len(character_classes)

    # Writer-class count. Prefer an explicit override; else read it off the UNet
    # checkpoint's label_emb so the model is built to match whatever it was
    # trained on (single IAM = 339, merged = W), avoiding a load_state_dict
    # shape mismatch / out-of-range style-bank lookups.
    ckpt_path = f"{args.save_path}/models/ckpt.pt"
    unet_sd = torch.load(ckpt_path, map_location=args.device, weights_only=True)
    override = int(getattr(args, "style_classes", 0) or 0)
    style_classes = override or _embedding_rows(unet_sd, "label_emb.weight") or 339
    print("style_classes (writer count):", style_classes)

    idx = int("".join(filter(str.isdigit, args.device)))
    device_ids = [idx, idx + 1] if args.dataparallel else [idx]

    transform = transforms.Compose([
        transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    tokenizer = CanineTokenizer.from_pretrained(
        "google/canine-c", local_files_only=_CANINE_LOCAL_ONLY
    )
    text_encoder = CanineModel.from_pretrained(
        "google/canine-c", local_files_only=_CANINE_LOCAL_ONLY
    )
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
            # Fail loud on a bank that does not match this checkpoint, unless the
            # caller opted in to a size-mismatched bank (--allow-bank-mismatch),
            # e.g. one built from a held-out/external split. Writer identity at
            # inference is bank-only (unet's label_emb branch is dead) and
            # get_style_coll range-checks --writer-id against the bank's own row
            # count, so a differing writer count is safe there -- the strict guard
            # only exists to catch an accidentally stale bank.
            if bank.shape[0] != style_classes:
                msg = (
                    "style bank writer count {} != model writer count {} ({}); the "
                    "bank was built from a different split -- rebuild it with "
                    "utils/build_style_bank.py against the training --data-dir".format(
                        bank.shape[0], style_classes, bank_path
                    )
                )
                if getattr(args, "allow_bank_mismatch", False):
                    print("WARNING (--allow-bank-mismatch):", msg)
                else:
                    raise ValueError(msg)
            slin = next(
                (v for k, v in unet_sd.items() if k.endswith("style_lin.weight")),
                None,
            )
            if slin is not None and bank.shape[1] != slin.shape[1]:
                raise ValueError(
                    "style bank feature dim {} != model style_lin input {}; the bank "
                    "was built with a different --style-name than the checkpoint was "
                    "trained with (e.g. mobilenetv2_100=1280 vs resnet18=512) -- "
                    "rebuild it with the matching --style-name".format(
                        bank.shape[1], slin.shape[1]
                    )
                )
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
        model_name=getattr(args, "style_name", "mobilenetv2_100"),
        num_classes=0,
        pretrained=True,
        trainable=True,
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

    unet.load_state_dict(unet_sd)  # loaded above to size style_classes
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
        placer_sd = torch.load(placer_path, map_location=args.device, weights_only=True)
        # Size the placer from its own checkpoint: writer count from the
        # line_advance buffer (per-writer, may differ from the UNet's -- placer
        # trains on IAM-only paragraph data), style_dim from the style_proj
        # input (style-only conditioning; there is no writer_emb anymore).
        placer_writers = _embedding_rows(placer_sd, "line_advance") or style_classes
        style_proj_w = next(
            (v for k, v in placer_sd.items() if k.endswith("style_proj.0.weight")), None
        )
        placer_style_dim = int(style_proj_w.shape[1]) if style_proj_w is not None else 1280
        # Fail loud if the loaded bank's feature dim disagrees with the placer's
        # (wrong --style-name: the placer and the bank must share one encoder).
        if (
            diffusion.style_bank is not None
            and diffusion.style_bank.shape[1] != placer_style_dim
        ):
            raise ValueError(
                "style bank feature dim {} != placer style_dim {}; the placer was "
                "trained against a bank from a different --style-name -- rebuild the "
                "bank / retrain the placer with the matching encoder".format(
                    diffusion.style_bank.shape[1], placer_style_dim
                )
            )
        placer = WordPlacer(num_writers=placer_writers, style_dim=placer_style_dim)
        placer = DataParallel(placer, device_ids=device_ids)
        placer.load_state_dict(placer_sd)
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
