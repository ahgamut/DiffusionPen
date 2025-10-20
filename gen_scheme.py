import argparse
import os
import pandas as pd
from PIL import Image, ImageOps
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
from PIL import Image
import skimage.filters as skfilt

#
from models import UNetModel, ImageEncoder
from models.diffpen2 import Diffusion, IAM_TempLoader
from utils.auxilary_functions import *
from utils.generation import (
    setup_logging,
    build_fake_image_N,
    build_fake_interp_N,
    add_rescale_padding,
    build_paragraph_image,
)
from utils.relcharsize import build_placed_paragraph
from utils.subprompt import Word, Prompt
from utils.arghandle import add_common_args


class CTX:
    mldict = dict()


def save_threshed(img, fname):
    # thresh separately? use Otsu?
    # thresh here?
    arr = np.array(img.convert("L"))
    thr = np.array(arr > skfilt.threshold_otsu(arr), dtype=np.uint8)
    thr = 255 * thr
    timg = Image.fromarray(thr).convert("L")
    timg.save(fname)


def resave_real(xmlname, imgname, targname):
    prompt = Prompt(xmlname)
    img = Image.open(imgname).convert("RGB")
    crop = prompt.get_cropped(img)
    save_threshed(crop, targname)


def build_ref_paragraph(fakes, xpr, max_line_width, longest_word_length):
    assert len(xpr.words) == len(fakes)
    dupe = Image.new("RGB", size=(xpr.img_width, xpr.img_height), color="white")

    for i in range(len(fakes)):
        word = xpr.words[i]
        fake = fakes[i]
        ratio = word.height / fake.height
        #
        scaled_width = int(fake.width * ratio)
        scaled_height = word.height
        scaled_width = max(scaled_width, 3)
        scaled_height = max(scaled_height, 3)
        scaled_img = fakes[i].resize((scaled_width, scaled_height))
        dupe.paste(scaled_img, (word.x_start, word.y_start))

    dupe = dupe.convert("L")
    return xpr.get_cropped(dupe)


def make_closedset(fname, targdir):
    df = pd.read_csv(fname)
    for ind, row in df.iterrows():
        wid = row["wid"].replace('"', "")
        xmlname = os.path.join("./iam_data/xml/", row["xmlname"])
        imgname = os.path.join("./iam_data/forms", row["imgname"])
        targname = os.path.join(targdir, row["target_name"]) + ".png"
        resave_real(xmlname, imgname, targname)


def resave_fake(xmlname, imgname, targname, faketype):
    xpr = Prompt(xmlname)
    raw_orig = Image.open(os.path.join("./iam_data", "forms", xpr.idd + ".png"))
    s = IAM_TempLoader.map_wid_to_index(xpr.writer_id)
    if "niceplace" in faketype:
        print("should regenerate", imgname, "place nicely and save")
        words = [w.raw for w in xpr.words]
        longest_word_length = max(len(word) for word in words)
        raw_crop = xpr.get_cropped(raw_orig)
        max_line_width = raw_crop.width
        max_word_length_width = 0
        fakes, max_word_length_width = build_fake_image_N(
            words,
            s,
            longest_word_length=longest_word_length,
            max_word_length_width=max_word_length_width,
            **CTX.mldict,
        )
        regen_img = build_ref_paragraph(
            fakes,
            xpr,
            max_line_width=max_line_width,
            longest_word_length=longest_word_length,
        )
        save_threshed(regen_img, targname)
        return

    if "traintext" in faketype:
        print("should regenerate", imgname, "place however and save")
        words = [w.raw for w in xpr.words]
        longest_word_length = max(len(word) for word in words)
        max_line_width = CTX.mldict["args"].max_line_width
        max_word_length_width = 0
        fakes, max_word_length_width = build_fake_image_N(
            words,
            s,
            longest_word_length=longest_word_length,
            max_word_length_width=max_word_length_width,
            **CTX.mldict,
        )
    elif "difftext1" in faketype:
        print("should generate LL using wid from", imgname, "and save")
        lines = open("./prompts/london-letter.txt").read()
        words = lines.strip().split(" ")
        longest_word_length = max(len(word) for word in words)
        max_line_width = CTX.mldict["args"].max_line_width
        max_word_length_width = 0
        fakes, max_word_length_width = build_fake_image_N(
            words,
            s,
            longest_word_length=longest_word_length,
            max_word_length_width=max_word_length_width,
            **CTX.mldict,
        )
    elif "difftext2" in faketype:
        print("should generate WOZ using wid from", imgname, "and save")
        lines = open("./prompts/woz-letter.txt").read()
        words = lines.strip().split(" ")
        longest_word_length = max(len(word) for word in words)
        max_line_width = CTX.mldict["args"].max_line_width
        max_word_length_width = 0
        fakes, max_word_length_width = build_fake_image_N(
            words,
            s,
            longest_word_length=longest_word_length,
            max_word_length_width=max_word_length_width,
            **CTX.mldict,
        )

    postparts = faketype.split("-")
    if len(postparts) == 2:
        scaled_padded_words = add_rescale_padding(
            words,
            fakes,
            max_word_length_width=max_word_length_width,
            longest_word_length=longest_word_length,
        )
        regen_img = build_paragraph_image(
            scaled_padded_words, max_line_width=max_line_width
        )
    elif len(postparts) == 4:
        font_size = int(postparts[2])
        use_aspect = postparts[3] == "img"
        regen_img = build_placed_paragraph(
            words,
            fakes,
            max_line_width=max_line_width,
            font_size=font_size,
            dpi=600,
            use_aspect=use_aspect,
        )
    else:
        raise RuntimeError("invalid post-processing:" + faketype)

    save_threshed(regen_img, targname)


def resave_interp(xmlname, imgname, targname, widinfo, interp):
    wid1, wid2, alpha = widinfo.split("-")
    wid1 = wid1.replace('"', "")
    wid2 = wid2.replace('"', "")
    alpha = float(alpha)
    if "sametext" in interp:
        print(
            "should interpolate between",
            (wid1, wid2),
            "at",
            alpha,
            "use same text and save to",
            targname,
        )
        xpr = Prompt(xmlname)
        words = [w.raw for w in xpr.words]
        longest_word_length = max(len(word) for word in words)
        raw_orig = Image.open(os.path.join("./iam_data", "forms", xpr.idd + ".png"))
        raw_crop = xpr.get_cropped(raw_orig)
        s1 = IAM_TempLoader.map_wid_to_index(wid1)
        s2 = IAM_TempLoader.map_wid_to_index(wid2)
        max_line_width = raw_crop.width
        CTX.mldict["args"].writer_1 = s1
        CTX.mldict["args"].writer_2 = s2
        CTX.mldict["args"].mix_rate = alpha
        max_word_length_width = 0
        fakes, max_word_length_width = build_fake_interp_N(
            words,
            longest_word_length=longest_word_length,
            max_word_length_width=max_word_length_width,
            **CTX.mldict,
        )
        regen_img2 = build_ref_paragraph(
            fakes,
            xpr,
            max_line_width=max_line_width,
            longest_word_length=longest_word_length,
        )
        save_threshed(regen_img2, targname)
    else:
        print(
            "should interpolate between",
            (wid1, wid2),
            "at",
            alpha,
            "use different text and save to",
            targname,
        )
        xpr = Prompt(xmlname)
        lines = open("./prompts/london-letter.txt").read()
        words = lines.strip().split(" ")
        longest_word_length = max(len(word) for word in words)
        s1 = IAM_TempLoader.map_wid_to_index(wid1)
        s2 = IAM_TempLoader.map_wid_to_index(wid2)
        max_line_width = CTX.mldict["args"].max_line_width
        CTX.mldict["args"].writer_1 = s1
        CTX.mldict["args"].writer_2 = s2
        CTX.mldict["args"].mix_rate = alpha
        max_word_length_width = 0
        fakes, max_word_length_width = build_fake_interp_N(
            words,
            longest_word_length=longest_word_length,
            max_word_length_width=max_word_length_width,
            **CTX.mldict,
        )
        regen_img = build_placed_paragraph(
            words,
            fakes,
            max_line_width=max_line_width,
            font_size=16,
            dpi=600,
            use_aspect=random.random() < 0.5,
        )
        save_threshed(regen_img, targname)

def process_csv(fname, targdir):
    print("processing", fname)
    if "clref" in fname:
        make_closedset(fname, targdir)
        return
    #
    df = pd.read_csv(fname)
    for ind, row in df.iterrows():
        wid = row["file2_wid"].replace('"', "")
        proc_tp = row["file2_type"]
        targname = os.path.join(targdir, row["target_name"]) + ".png"

        if proc_tp == "real":
            img_basename = row["file2_path"]
            imgname = os.path.join("./iam_data/forms", img_basename)
            xmlname = os.path.join("./iam_data/xml", img_basename.replace("png", "xml"))
            resave_real(xmlname, imgname, targname)

        elif proc_tp.startswith("fake-"):
            img_basename = row["file2_path"]
            imgname = os.path.join("./iam_data/forms", img_basename)
            xmlname = os.path.join("./iam_data/xml", img_basename.replace("png", "xml"))
            resave_fake(xmlname, imgname, targname, proc_tp)

        else:
            anchor_basename = row["file1_path"]
            imgname = os.path.join("./iam_data/forms", anchor_basename)
            xmlname = os.path.join(
                "./iam_data/xml", anchor_basename.replace("png", "xml")
            )
            widinfo = row["file2_path"]
            resave_interp(xmlname, imgname, targname, widinfo, proc_tp)


def main():
    parser = argparse.ArgumentParser("generate-scheme")
    parser.add_argument(
        "--config-dir",
        default="./saved_iam_data",
        help="file containing config CSVs",
    )
    parser.add_argument("--output-dir", default="./saved_iam_data", help="output dir")
    parser.add_argument(
        "--max-line-width", default=900, type=int, help="max line width"
    )
    add_common_args(parser)
    args = parser.parse_args()
    ####
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
    ####

    CTX.mldict["diffusion"] = diffusion
    CTX.mldict["ema_model"] = ema_model
    CTX.mldict["vae"] = vae
    CTX.mldict["feature_extractor"] = feature_extractor
    CTX.mldict["ddim"] = ddim
    CTX.mldict["transform"] = transform
    CTX.mldict["tokenizer"] = tokenizer
    CTX.mldict["text_encoder"] = text_encoder
    CTX.mldict["args"] = args
    IAM_TempLoader.check_preload()

    ####
    pieces = ["clref", "qmreal", "qnreal", "qmfake", "qnfake", "qinterp"]
    for p in pieces:
        fname = os.path.join(args.config_dir, f"samp-{p}.csv")
        targdir = os.path.join(args.output_dir, p)
        os.makedirs(targdir, exist_ok=True)
        process_csv(fname, targdir)


if __name__ == "__main__":
    main()
