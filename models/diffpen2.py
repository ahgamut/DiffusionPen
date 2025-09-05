import os
import torch
from PIL import Image, ImageOps
import json
import random
from torchvision import transforms

#
from .auxiliary_functions import (
    image_resize_PIL,
    centered_PIL,
)


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
    def get_refs(cls, label_index, n_samples):
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

    def get_style(
        self,
        label_index,
        transform,
        args,
        temp_loader,
        interpol=False,
        cor_im=False,
    ):
        #
        fheight, fwidth = 64, 256
        device = args.device
        #
        five_refs = temp_loader.get_refs(label_index, 5)
        # print("five_styles", five_refs["paths"])

        cor_image_random = temp_loader.get_refs(label_index, 1)
        if cor_im:
            cor_image = cor_image_random["imgs"][0]
            cor_image = iam_resizefix(cor_image)
            cor_im_tens = transform(cor_image).to(device)
            cor_im_tens = cor_im_tens.unsqueeze(0)
            cor_images = vae.module.encode(
                cor_im_tens.to(torch.float32)
            ).latent_dist.sample()
            cor_images = cor_images * 0.18215

        st_imgs = []
        for im_idx, img_s in enumerate(five_refs["imgs"]):
            img_s = iam_resizefix(img_s)
            img_tens = transform(img_s).to(device)  # .unsqueeze(0)
            st_imgs += [img_tens]

        # save style images
        style_images = torch.stack(st_imgs).to(device)
        style_images = style_images.reshape(-1, 3, 64, 256)
        return style_images

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
        cor_im = False
        interpol = False
        temp_loader = None

        if args.dataset == "iam":
            temp_loader = IAM_TempLoader
        temp_loader.check_preload()

        with torch.no_grad():
            text_features = x_text
            text_features = tokenizer(
                text_features,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=40,
            ).to(args.device)

            style_coll = {"images": [], "features": []}

            if args.img_feat:
                for label in labels:
                    label_index = label.item()
                    s_imgs = self.get_style(
                        label_index,
                        transform,
                        args,
                        temp_loader,
                        cor_im=cor_im,
                        interpol=interpol,
                    )
                    s_feat = style_extractor(s_imgs).to(args.device)
                    style_coll["images"].append(s_imgs)
                    style_coll["features"].append(s_feat)

                style_images = torch.cat(style_coll["images"])
                style_features = torch.cat(style_coll["features"])
            else:
                style_images = None
                style_features = None

            #
            if args.latent:
                x = torch.randn(
                    (n, 4, self.img_size[0] // 8, self.img_size[1] // 8)
                ).to(args.device)
                if cor_im:
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
                noisy_residual = model(
                    x,
                    timesteps=t,
                    context=text_features,
                    y=labels,
                    original_images=style_images,
                    mix_rate=mix_rate,
                    style_extractor=style_features,
                )
                prev_noisy_sample = noise_scheduler.step(
                    noisy_residual, time, x
                ).prev_sample
                x = prev_noisy_sample

        model.train()
        if args.latent:
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

    def interp_sampling(
        self,
        model,
        vae,
        n,
        x_text,
        labels,
        weight,
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
        assert len(labels) == 2
        n = 1

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
            if args.latent == True:
                x = torch.randn(
                    (n, 4, self.img_size[0] // 8, self.img_size[1] // 8)
                ).to(args.device)
            else:
                x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(
                    args.device
                )

            # scheduler
            noise_scheduler.set_timesteps(50)
            for time in noise_scheduler.timesteps:
                t_item = time.item()
                t = (torch.ones(n) * t_item).long().to(args.device)
                noisy_residual = model(
                    x=x,
                    s1=labels[0].item(),
                    s2=labels[1].item(),
                    interpolation=True,
                    mix_rate=args.interp_weight,
                    timesteps=t,
                    context=text_features,
                    original_images=None,
                    style_extractor=None,
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
