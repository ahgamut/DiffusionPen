import glob
import sys
import traceback
import random
import os
import torch
import torch.nn as nn
import torchvision
from torch import optim
import copy
import argparse
from diffusers import AutoencoderKL, DDIMScheduler
from torch.nn import DataParallel
from torchvision import transforms
from transformers import CanineModel, CanineTokenizer
from PIL import Image, ImageDraw

#
from models import UNetModel, ImageEncoder
from models.diffpen2 import Diffusion, IAM_TempLoader
from utils.auxilary_functions import *
from utils.generation import (
    setup_logging,
    build_fake_image,
    add_rescale_padding,
    build_paragraph_image,
)
from utils.arghandle import add_common_args
from utils.subprompt import Prompt as XMLPrompt

OUTPUT_MAX_LEN = 95  # + 2  # <GO>+groundtruth+<END>
IMG_WIDTH = 256
IMG_HEIGHT = 64
PUNCTUATION = "_!\"#&'()*+,-./:;?"


def file_check(fname):
    if os.path.isfile(fname):
        return fname
    raise RuntimeError(f"{fname} is not a file")


def build_ref_paragraph(fakes, xpr, max_line_width, longest_word_length):
    assert len(xpr.words) == len(fakes)
    dupe = Image.new("RGB", size=(xpr.width, xpr.height), color="white")

    for i in range(len(fakes)):
        word = xpr.words[i]
        fake = fakes[i]
        ratio = word.height / fake.height
        #
        scaled_width = int(fake.width * ratio)
        scaled_height = word.height
        scaled_img = fakes[i].resize((scaled_width, scaled_height))
        dupe.paste(scaled_img, (word.x_start, word.y_start))

    dupe = dupe.convert("L")
    return dupe


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
        if len(word) == longest_word_length:
            max_word_length_width = im.width
        fakes.append(im)
    return fakes, max_word_length_width


def load_prompt(coll):
    xpr = None
    fname = None
    try:
        fname = random.choice(coll)
        xpr = XMLPrompt(fname)
        assert xpr.writer_id in IAM_TempLoader.wr_dict
    except Exception:
        print(f"failed to read {fname}")
        tb = traceback.format_tb(sys.exc_info()[2])
        print("".join(tb))
        xpr = None
    return xpr


def main():
    parser = argparse.ArgumentParser("regen-prompts")
    parser.add_argument(
        "-n", "--num-prompts", default=1, type=int, help="number of prompts"
    )
    parser.add_argument("-o", "--output", type=str, default="./outputs/")
    parser.add_argument("--alt-text", default="./prompts/sample.txt", help="alt text")
    add_common_args(parser)

    args = parser.parse_args()
    print("torch version", torch.__version__)

    # create save directories
    setup_logging(args)
    IAM_TempLoader.check_preload()
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

    coll_xmls = list(glob.glob("./iam_data/xml/*.xml"))
    alt_lines = open(args.alt_text).read()
    alt_words = alt_lines.strip().split(" ")

    print("duplicating prompt")
    for i in range(args.num_prompts):
        try:
            xpr = load_prompt(coll_xmls)
            while xpr is None:
                xpr = load_prompt(coll_xmls)
            raw_orig = Image.open(os.path.join("./iam_data", "forms", xpr.id + ".png"))
            raw_crop = xpr.get_cropped(raw_orig)
            s = IAM_TempLoader.map_wid_to_index(xpr.writer_id)
            max_line_width = raw_crop.width

            # same prompt
            words = [w.raw for w in xpr.words]
            fakes = []
            max_word_length_width = 0
            longest_word_length = max(len(word) for word in words)

            fakes, max_word_length_width = build_fakes(
                words,
                s,
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
            scaled_padded_words = add_rescale_padding(
                words,
                fakes,
                max_word_length_width=max_word_length_width,
                longest_word_length=longest_word_length,
            )
            regen_img = build_paragraph_image(
                scaled_padded_words, max_line_width=max_line_width
            )
            regen_img2 = build_ref_paragraph(
                fakes,
                xpr,
                max_line_width=max_line_width,
                longest_word_length=longest_word_length,
            )

            #
            words = alt_words
            fakes = []
            max_word_length_width = 0
            longest_word_length = max(len(word) for word in words)
            fakes, max_word_length_width = build_fakes(
                words,
                s,
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
            scaled_padded_words = add_rescale_padding(
                words,
                fakes,
                max_word_length_width=max_word_length_width,
                longest_word_length=longest_word_length,
            )
            regen_alt = build_paragraph_image(
                scaled_padded_words, max_line_width=max_line_width
            )

            #
            rid = "%04x" % random.randint(0, 1000)
            raw_crop.save(os.path.join(args.output, f"{xpr.id}_orig.png"))
            regen_img.save(os.path.join(args.output, f"{xpr.id}_fake_{rid}.png"))
            regen_img2.save(os.path.join(args.output, f"{xpr.id}_fake-sz_{rid}.png"))
            regen_alt.save(os.path.join(args.output, f"{xpr.id}_alt_{rid}.png"))
        except Exception as e:
            print(e)
            tb = traceback.format_tb(sys.exc_info()[2])
            print("".join(tb))


if __name__ == "__main__":
    main()
