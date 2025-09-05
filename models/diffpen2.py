import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps
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
from .auxiliary_functions import (
    affine_transformation,
    image_resize,
    image_resize_PIL,
    centered,
    centered_PIL,
)
from .feature_extractor import ImageEncoder
from .diffpen import AvgMeter, EMA


def iam_resizefix(img_s):
    (img_width, img_height) = img_s.size
    img_s = img_s.resize((int(img_width * 64 / img_height), 64))
    (img_width, img_height) = img_s.size

    if img_width < 256:
        outImg = ImageOps.pad(
            img_s, size=(256, 64), color="white"
        )  # , centering=(0,0)) uncommment to pad right
        img_s = outImg

    else:
        # reduce image until width is smaller than 256
        while img_width > 256:
            img_s = image_resize_PIL(img_s, width=img_width - 20)
            (img_width, img_height) = img_s.size
        img_s = centered_PIL(img_s, (64, 256), border_value=255.0)

    return img_s


class IAM_TempLoader:

    wr_dict = None
    reverse_wr_dict = None
    train_data = None
    root_path = "./iam_data/words"
    wmap = None
    tform = None

    @classmethod
    def check_preload(cls):
        if cls.wr_dict is None:
            with open("utils/writers_dict_train_iam.json", "r") as f:
                cls.wr_dict = json.load(f)
            cls.reverse_wr_dict = {v: k for k, v in cls.wr_dict.items()}

        if cls.train_data is None:
            with open("./utils/splits_words/iam_train_val.txt", "r") as f:
                # with open('./utils/splits_words/iam_test.txt', 'r') as f:
                train_data = f.readlines()
                cls.train_data = [i.strip().split(",") for i in train_data]

        if cls.wmap is None:
            wmap = dict()
            for obj in cls.train_data:
                img_path = obj[0]
                wid = obj[1]
                transcr = ",".join(obj[2:])
                if wid in wmap.keys():
                    wmap[wid].append((img_path, wid, transcr))
                else:
                    wmap[wid] = [(img_path, wid, transcr)]
            cls.wmap = wmap

        if cls.tform is None:
            cls.tform = transforms.ToTensor()

    @classmethod
    def get_styles(cls, label_index, n_samples):
        wid = cls.reverse_wr_dict[label_index]
        matching_lines = cls.wmap[wid]

        paths = []
        imgs = []
        while len(imgs) < 5:
            mas = random.sample(matching_lines, n_samples)
            for ma in mas:
                ma_path = None
                ma_img = None
                if len(ma[2]) > 3:
                    ma_path = os.path.join(cls.root_path, ma[0])
                if ma_path is not None:
                    try:
                        ma_img = Image.open(ma_path).convert("RGB")
                    except Exception:
                        # Handle the exception (e.g., print an error message)
                        print(f"Error loading image from {ma_path}")
                if ma_img is not None:
                    imgs.append(ma_img)
                    paths.append(ma[0])

        result = {"paths": paths[:5], "imgs": imgs[:5]}
        return result

    @classmethod
    def load(cls, label, vae, args, interpol=False, cor_im=False):
        result = dict()
        # print('label', label)
        cls.check_preload()
        # pick random image according to specific style
        style_featur = []
        label_index = label.item()

        if interpol:
            label2 = random.randint(0, 339)  # random label
            five_styles = cls.get_styles(label2, 5)
        else:
            five_styles = cls.get_styles(label_index, 5)

        print("five_styles", five_styles["paths"])
        # cor_image
        fheight, fwidth = 64, 256
        if cor_im == True:
            cor_image_random = cls.get_styles(label_index, 1)["imgs"]
            cor_image = Image.open(
                os.path.join(cls.root_path, cor_image_random[0])
            ).convert("RGB")
            cor_image = iam_resizefix(cor_image)
            cor_im_tens = transform(cor_image).to(args.device)
            # print('cor image', cor_im_tens.shape)
            cor_im_tens = cor_im_tens.unsqueeze(0)
            cor_images = vae.module.encode(
                cor_im_tens.to(torch.float32)
            ).latent_dist.sample()
            cor_images = cor_images * 0.18215
            result["cor_images"] = cor_images

        st_imgs = []
        grid_imgs = []
        for im_idx, img_s in enumerate(five_styles["imgs"]):
            img_s = iam_resizefix(img_s)
            # make grid of all 5 images
            # img_s = img_s.convert('L')
            grid_im = cls.tform(img_s)
            grid_imgs += [grid_im]
            img_tens = cls.tform(img_s).to(args.device)  # .unsqueeze(0)
            st_imgs += [img_tens]

        grid_imgs = torch.stack(grid_imgs).to(args.device)
        style_images = torch.stack(st_imgs).to(args.device)

        result["grid_imgs"] = grid_imgs
        result["style_images"] = style_images
        return result


class Diffusion:
    def __init__(
        self,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=(64, 256),
        args=None,
    ):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(args.device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = args.device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sampling(
        self,
        model,
        vae,
        n,
        x_text,
        labels,
        args,
        style_extractor,
        noise_scheduler,
        mix_rate=None,
        cfg_scale=3,
        transform=None,
        character_classes=None,
        tokenizer=None,
        text_encoder=None,
        run_idx=None,
    ):
        model.eval()
        tensor_list = []

        with torch.no_grad():
            text_features = x_text  # [x_text]*n
            # print('text features', text_features.shape)
            text_features = tokenizer(
                text_features,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=40,
            ).to(args.device)
            if args.dataset == "iam":
                temp_loader = IAM_TempLoader
            elif args.dataset == "cvl":
                temp_loader = CVL_TempLoader
            else:
                temp_loader = None
            if args.img_feat == True:
                cor_im = False
                for label in labels:
                    stuff = temp_loader.load(label, vae, args, interpol=False, cor_im=False)
                    style_images = stuff["style_images"].reshape(-1, 3, 64, 256)
                    style_features = style_extractor(style_images).to(args.device)
                    print(style_images.shape, style_features.shape)
            else:
                style_images = None
                style_features = None
            if args.latent == True:
                x = torch.randn(
                    (n, 4, self.img_size[0] // 8, self.img_size[1] // 8)
                ).to(args.device)
                if cor_im == True:
                    x_noise = torch.randn(cor_images.shape).to(args.device)

                    timesteps = torch.full(
                        (cor_images.shape[0],),
                        999,
                        device=args.device,
                        dtype=torch.long,
                    )

                    noisy_images = noise_scheduler.add_noise(
                        cor_images, x_noise, timesteps
                    )
                    x = noisy_images

            else:
                x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(
                    args.device
                )

            # scheduler
            noise_scheduler.set_timesteps(50)
            for time in noise_scheduler.timesteps:

                t_item = time.item()
                t = (torch.ones(n) * t_item).long().to(args.device)

                with torch.no_grad():
                    noisy_residual = model(
                        x,
                        t,
                        text_features,
                        labels,
                        original_images=style_images,
                        mix_rate=mix_rate,
                        style_extractor=style_features,
                    )
                    prev_noisy_sample = noise_scheduler.step(
                        noisy_residual, time, x
                    ).prev_sample
                    x = prev_noisy_sample

        model.train()
        if args.latent == True:
            latents = 1 / 0.18215 * x
            image = vae.module.decode(latents).sample

            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()

            image = torch.from_numpy(image)
            x = image.permute(0, 3, 1, 2)

        else:
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
        return x
