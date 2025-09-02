import os
import sys
import traceback
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
from torchvision import transforms
from torchvision.utils import save_image
from torch.nn import DataParallel
from transformers import CanineModel, CanineTokenizer

#
from models import EMA, Diffusion, UNetModel, ImageEncoder
from utils.iam_dataset import IAMDataset
from utils.GNHK_dataset import GNHK_Dataset
from utils.auxilary_functions import *

torch.cuda.empty_cache()
OUTPUT_MAX_LEN = 95  # + 2  # <GO>+groundtruth+<END>
IMG_WIDTH = 256
IMG_HEIGHT = 64

PUNCTUATION = "_!\"#&'()*+,-./:;?"


def setup_logging(args):
    # os.makedirs("models", exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "images"), exist_ok=True)


def save_images(images, path, args, **kwargs):
    # print('image', images.shape)
    grid = torchvision.utils.make_grid(images, padding=0, **kwargs)
    if args.latent == True:
        im = torchvision.transforms.ToPILImage()(grid)
        if args.color == False:
            im = im.convert("L")
        else:
            im = im.convert("RGB")
    else:
        ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
        im = Image.fromarray(ndarr)
    im.save(path)
    return im


def crop_whitespace_width(img):
    # tensor image to PIL
    original_height = img.height
    img_gray = np.array(img)
    ret, thresholded = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    coords = cv2.findNonZero(thresholded)
    x, y, w, h = cv2.boundingRect(coords)
    # rect = img.crop((x, 0, x + w, original_height))
    rect = img.crop((x, y, x + w, y + h))
    return np.array(rect)


def file_check(fname):
    if os.path.isfile(fname):
        return fname
    raise RuntimeError(f"{fname} is not a file")


def range_check(x):
    l, u = x.split("-")
    l = int(l)
    u = int(u)

    if l < 0 or u < 0 or l > u:
        raise RuntimeError(f"invalid range: {x}")

    return (l, u)


def build_fakes(
    pieces,
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
    for word in pieces:
        print("Word:", word)
        labels = torch.tensor([s]).long().to(args.device)
        ema_sampled_images = diffusion.sampling(
            ema_model,
            vae,
            n=len(labels),
            x_text=word,
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
        image = ema_sampled_images.squeeze(0)
        im = torchvision.transforms.ToPILImage()(image)
        # reshape to height 32
        im = im.convert("L")
        # save im

        im = crop_whitespace_width(im)
        im = Image.fromarray(im)
        if len(word) == longest_word_length:
            max_word_length_width = im.width
            print("max_word_length_width", max_word_length_width)
        im = np.array(im)
        # im = np.array(resized_img)
        fakes.append(im)
    return fakes, max_word_length_width


def add_padding(
    pieces, fakes, max_word_length_width, longest_word_length, max_height=64
):
    # find the average character width of the max length word
    avg_char_width = max_word_length_width / longest_word_length
    print("avg_char_width", avg_char_width)
    scaled_padded_words = []

    for word, img in zip(pieces, fakes):
        img_pil = Image.fromarray(img)
        as_ratio = img_pil.width / img_pil.height
        # scaled_width = int(scaling_factor * len(word))#) * as_ratio * max_height)
        scaled_width = max(5, int(avg_char_width * len(word)))
        scaled_height = max(5, int(scaled_width / as_ratio))

        scaled_img = img_pil.resize((scaled_width, scaled_height))
        print(f"Word {word} - scaled_img {scaled_img.size}")
        # Padding
        # if word is in PUNCTUATION:
        if word in PUNCTUATION:
            # rescale to height 10
            w_punc = scaled_img.width
            h_punc = scaled_img.height
            as_ratio_punct = w_punc / h_punc
            if word == ".":
                scaled_img = scaled_img.resize((int(5 * as_ratio_punct), 5))
            else:
                scaled_img = scaled_img.resize((int(13 * as_ratio_punct), 13))
            # pad on top and leave the image in the bottom
            padding_bottom = 10
            padding_top = (
                max_height - scaled_img.height - padding_bottom
            )  # All padding goes on top
            # No padding at the bottom

            # Apply padding
            padded_img = np.pad(
                scaled_img,
                ((padding_top, padding_bottom), (0, 0)),
                mode="constant",
                constant_values=255,
            )
        else:
            if scaled_img.height < max_height:
                padding = (max_height - scaled_img.height) // 2
                # print(f'Word {word} - padding: {padding}')
                padded_img = np.pad(
                    scaled_img,
                    (
                        (padding, max_height - scaled_img.height - padding),
                        (0, 0),
                    ),
                    mode="constant",
                    constant_values=255,
                )
            else:
                # resize to max height while maintaining aspect ratio
                # ar = scaled_img.width / scaled_img.height

                scaled_img = scaled_img.resize(
                    (int(max_height * as_ratio) - 4, max_height - 4)
                )
                padding = (max_height - scaled_img.height) // 2
                # print(f'Word {word} - padding: {padding}')
                padded_img = np.pad(
                    scaled_img,
                    (
                        (padding, max_height - scaled_img.height - padding),
                        (0, 0),
                    ),
                    mode="constant",
                    constant_values=255,
                )

            # padded_img = np.array(scaled_img)
        # print('padded_img', padded_img.shape)
        scaled_padded_words.append(padded_img)
    return scaled_padded_words


def build_paragraph_image(
    scaled_padded_words, max_line_width=900, gap_height=64, gap_width=16
):
    gap = np.ones((gap_height, gap_width), dtype=np.uint8) * 255  # White gap
    current_line_width = 0
    # Concatenate images with gaps
    sentence_img = gap  # Start with a gap
    lines = []
    line_img = gap
    # Concatenate images with gaps
    """
    sentence_img = gap  # Start with a gap
    for img in scaled_padded_words:
        #print('img', img.shape)
        sentence_img = np.concatenate((sentence_img, img, gap), axis=1)
    """

    for img in scaled_padded_words:
        img_width = img.shape[1] + gap.shape[1]

        if current_line_width + img_width < max_line_width:
            # Add the image to the current line
            if line_img.shape[0] == 0:
                line_img = (
                    np.ones((gap_height, 0), dtype=np.uint8) * 255
                )  # Start a new line
            line_img = np.concatenate((line_img, img, gap), axis=1)
            current_line_width += img_width  # + gap.shape[1]
            # print('current_line_width if', current_line_width)
            # Check if adding this image exceeds the max line width
        else:
            # Pad the current line with white space to max_line_width
            remaining_width = max_line_width - current_line_width
            line_img = np.concatenate(
                (
                    line_img,
                    np.ones((gap_height, remaining_width), dtype=np.uint8) * 255,
                ),
                axis=1,
            )
            lines.append(line_img)

            # Start a new line with the current word
            line_img = np.concatenate((gap, img, gap), axis=1)
            current_line_width = img_width  # + 2 * gap.shape[1]
            # print('current_line_width else', current_line_width)
    # Add the last line to the lines list
    if current_line_width > 0:
        # Pad the last line to max_line_width
        remaining_width = max_line_width - current_line_width
        line_img = np.concatenate(
            (
                line_img,
                np.ones((gap_height, remaining_width), dtype=np.uint8) * 255,
            ),
            axis=1,
        )
        lines.append(line_img)

    # # Concatenate all lines to form a paragraph, pad them if necessary
    # max_height = max([line.shape[0] for line in lines])
    # paragraph_img = np.ones((0, max_line_width), dtype=np.uint8) * 255
    # for line in lines:
    #     if line.shape[0] < max_height:
    #         padding = (max_height - line.shape[0]) // 2
    #         line = np.pad(line, ((padding, max_height - line.shape[0] - padding), (0, 0)), mode='constant', constant_values=255)

    #     #print the shapes
    #     print('line shape', line.shape)
    # print('paragraph shape', paragraph_img.shape)
    paragraph_img_raw = np.concatenate((lines), axis=0)
    paragraph_image = Image.fromarray(paragraph_img_raw)
    paragraph_image = paragraph_image.convert("L")
    return paragraph_image


def main():
    """Main function"""
    parser = argparse.ArgumentParser("diffusion-paragraph-bulk")
    parser.add_argument(
        "--model_name",
        type=str,
        default="diffusionpen",
        help="(deprecated)",
    )
    parser.add_argument("-w", "--writer-range", type=range_check, default=(1, 1))
    parser.add_argument("--level", type=str, default="word", help="word, line")
    parser.add_argument("--img-size", type=int, default=(64, 256))
    parser.add_argument(
        "--max-line-width", default=900, type=int, help="max line width"
    )
    # UNET parameters
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--emb_dim", type=int, default=320)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_res_blocks", type=int, default=1)
    parser.add_argument(
        "--save_path", type=str, default="./diffusionpen_iam_model_path"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--color", type=bool, default=True)
    parser.add_argument("--latent", type=bool, default=True)
    parser.add_argument("--img_feat", type=bool, default=True)
    parser.add_argument("--interpolation", type=bool, default=False)
    parser.add_argument("--dataparallel", type=bool, default=False)
    parser.add_argument("--load_check", type=bool, default=False)
    parser.add_argument("--mix_rate", type=float, default=None)
    parser.add_argument(
        "--style_path", type=str, default="./style_models/iam_style_diffusionpen.pth"
    )
    parser.add_argument(
        "--stable_dif_path", type=str, default="./stable-diffusion-v1-5"
    )
    parser.add_argument("-i", "--text-file", type=file_check, default="./sample.txt")
    parser.add_argument("-o", "--output", type=str, default="./output.png")

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

    diffusion = Diffusion(img_size=args.img_size, args=args)

    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)

    # load from last checkpoint

    if args.load_check == True:
        unet.load_state_dict(
            torch.load(f"{args.save_path}/models/ckpt.pt", weights_only=True)
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

    # make the code to generate lines
    lines = open(args.text_file).read()
    pieces = lines.strip().split(" ")
    max_line_width = args.max_line_width
    max_word_length_width = 0
    longest_word_length = max(len(word) for word in pieces)

    output_template = args.output.replace(".png", "-{s}.png")
    writer_range = args.writer_range
    for s in range(writer_range[0], writer_range[1] + 1):
        print("Style:", s)
        try:
            # build fake words
            fakes, max_word_length_width = build_fakes(
                pieces,
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
            scaled_padded_words = add_padding(
                pieces,
                fakes,
                max_word_length_width=max_word_length_width,
                longest_word_length=longest_word_length,
            )

            # combine to create paragraph
            paragraph_image = build_paragraph_image(
                scaled_padded_words, max_line_width=max_line_width
            )
            paragraph_image.save(output_template.format(s=s))
        except Exception as e:
            print("failed for", s)
            print(e)
            tb = traceback.format_tb(sys.exc_info()[2])
            print("".join(tb))


if __name__ == "__main__":
    main()
