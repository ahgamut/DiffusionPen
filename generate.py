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
from models import EMA, Diffusion, UNetModel, ImageEncoder
from torchvision import transforms
from utils.iam_dataset import IAMDataset
from utils.GNHK_dataset import GNHK_Dataset
from utils.auxilary_functions import *
from torchvision.utils import save_image
from torch.nn import DataParallel
from transformers import CanineModel, CanineTokenizer


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


def main():
    """Main function"""
    parser = argparse.ArgumentParser("diffusionpen-singleword")
    parser.add_argument(
        "--model_name",
        type=str,
        default="diffusionpen",
        help="(deprecated)",
    )
    parser.add_argument("-w", "--writer-id", type=int, default=12)
    parser.add_argument("--level", type=str, default="word", help="word, line")
    parser.add_argument("--img_size", type=int, default=(64, 256))
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
    parser.add_argument("--sampling_word", type=str, default="hello")

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

    character_classes = [
        "!",
        '"',
        "#",
        "&",
        "'",
        "(",
        ")",
        "*",
        "+",
        ",",
        "-",
        ".",
        "/",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        ":",
        ";",
        "?",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        " ",
    ]

    ######################### MODEL #######################################
    vocab_size = len(character_classes)
    style_classes = 339 # for IAM Dataset
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
        unet.load_state_dict(torch.load(f"{args.save_path}/models/ckpt.pt"))
        optimizer.load_state_dict(torch.load(f"{args.save_path}/models/optim.pt"))
        ema_model.load_state_dict(torch.load(f"{args.save_path}/models/ema_ckpt.pt"))
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

    state_dict = torch.load(PATH, map_location=args.device)
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
        torch.load(f"{args.save_path}/models/ckpt.pt", map_location=args.device)
    )
    print("unet loaded")
    unet.eval()

    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
    ema_model.load_state_dict(torch.load(f"{args.save_path}/models/ema_ckpt.pt"))
    ema_model.eval()

    text_words = [args.sampling_word]
    # writer_id = random.randint(0, 339)
    writer_id = args.writer_id  # index for style class
    for x_text in text_words:
        print("Word:", x_text)
        print("style", writer_id)
        labels = torch.tensor([writer_id]).long().to(args.device)
        ema_sampled_images = diffusion.sampling(
            ema_model,
            vae,
            n=len(labels),
            x_text=x_text,
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
        save_single_images(
            ema_sampled_images,
            os.path.join(f"./image_samples/", f"{x_text}_style_{writer_id}.png"),
            args,
        )


if __name__ == "__main__":
    main()
