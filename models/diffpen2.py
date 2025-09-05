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
        label_index = label.item()

        if interpol:
            label2 = random.randint(0, 339)  # random label
            five_styles = cls.get_styles(label2, 5)
        else:
            five_styles = cls.get_styles(label_index, 5)

        print("five_styles", five_styles["paths"])
        # cor_image
        fheight, fwidth = 64, 256
        if cor_im:
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

    def get_style(
        self,
        label_index,
        train_data,
        wr_dict,
        reverse_wr_dict,
        transform,
        args,
        interpol=False,
        cor_im=False,
    ):
        fheight, fwidth = 64, 256
        root_path = "./iam_data/words"
        matching_lines = [
            line
            for line in train_data
            if line[1] == reverse_wr_dict[label_index] and len(line[2]) > 3
        ]

        # pick the first 5 from matching lines
        if len(matching_lines) >= 5:
            five_styles = random.sample(matching_lines, 5)
        else:
            matching_lines = [
                line for line in train_data if line[1] == reverse_wr_dict[label_index]
            ]
            five_styles = matching_lines_style[:5]
            if len(five_styles) < 5:
                five_styles = [matching_lines[0]] * 5
        print("five_styles", five_styles)

        cor_image_random = random.sample(matching_lines, 1)
        # print('five_styles', five_styles)
        # cor_image
        if cor_im:
            cor_image = Image.open(
                os.path.join(root_path, cor_image_random[0][0])
            ).convert("RGB")
            (cor_image_width, cor_image_height) = cor_image.size
            cor_image = cor_image.resize(
                (int(cor_image_width * 64 / cor_image_height), 64)
            )
            (cor_image_width, cor_image_height) = cor_image.size

            if cor_image_width < 256:
                outImg = ImageOps.pad(cor_image, size=(256, 64), color="white")
                cor_image = outImg

            else:
                # reduce image until width is smaller than 256
                while cor_image_width > 256:
                    cor_image = image_resize_PIL(cor_image, width=cor_image_width - 20)
                    (cor_image_width, cor_image_height) = cor_image.size
                cor_image = centered_PIL(cor_image, (64, 256), border_value=255.0)

            cor_im_tens = transform(cor_image).to(args.device)
            # print('cor image', cor_im_tens.shape)
            cor_im_tens = cor_im_tens.unsqueeze(0)
            cor_images = vae.module.encode(
                cor_im_tens.to(torch.float32)
            ).latent_dist.sample()
            cor_images = cor_images * 0.18215

        st_imgs = []
        grid_imgs = []
        for im_idx, random_f in enumerate(five_styles):
            file_path = os.path.join(root_path, random_f[0])
            try:
                img_s = Image.open(file_path).convert("RGB")
            except ValueError:
                # Handle the exception (e.g., print an error message)
                print(f"Error loading image from {file_path}")

                # Find a replacement image that is not corrupted
                replacement_idx = (im_idx + 1) % 5
                replacement_f = five_styles[replacement_idx]
                name = replacement_f[0]  # .split(',')[1]
                replacement_file_path = os.path.join(root_path, name)
                img_s = Image.open(replacement_file_path).convert("RGB")

            (img_width, img_height) = img_s.size
            img_s = img_s.resize((int(img_width * 64 / img_height), 64))
            (img_width, img_height) = img_s.size

            if img_width < 256:
                outImg = ImageOps.pad(img_s, size=(256, 64), color="white")
                img_s = outImg

            else:
                # reduce image until width is smaller than 256
                while img_width > 256:
                    img_s = image_resize_PIL(img_s, width=img_width - 20)
                    (img_width, img_height) = img_s.size
                img_s = centered_PIL(img_s, (64, 256), border_value=255.0)
            # make grid of all 5 images
            # img_s = img_s.convert('L')
            transform_tensor = transforms.ToTensor()
            grid_im = transform_tensor(img_s)
            grid_imgs += [grid_im]

            img_tens = transform(img_s).to(args.device)  # .unsqueeze(0)
            st_imgs += [img_tens]
            # style_features = style_extractor(style_images).to(args.device)
            # img_tensor = img_tensor.to(args.device)
        style_images = torch.stack(st_imgs).to(args.device)
        grid_imgs = torch.stack(grid_imgs).to(args.device)

        # save style images
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
                # pick random image according to specific style
                with open("utils/writers_dict_train.json", "r") as f:
                    wr_dict = json.load(f)
                    reverse_wr_dict = {v: k for k, v in wr_dict.items()}

                # key = reverse_wr_dict[value]
                with open("./utils/splits_words/iam_train_val.txt", "r") as f:
                    train_data = f.readlines()
                    train_data = [i.strip().split(",") for i in train_data]

                for label in labels:
                    label_index = label.item()
                    s_imgs = self.get_style(
                        label_index,
                        train_data,
                        wr_dict,
                        reverse_wr_dict,
                        transform,
                        args,
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
