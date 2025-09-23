import os
import torch
from PIL import Image, ImageOps
from tqdm import tqdm
import json
import random
from torchvision import transforms

#
from .auxiliary_functions import (
    image_resize_PIL,
    centered_PIL,
)


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


class EMA:
    """
    EMA is used to stabilize the training process of diffusion models by
    computing a moving average of the parameters, which can help to reduce
    the noise in the gradients and improve the performance of the model.
    """

    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


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

    def sampling_loader(
        self,
        model,
        test_loader,
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
    ):
        model.eval()

        with torch.no_grad():
            pbar = tqdm(test_loader)
            for i, data in enumerate(pbar):
                images = data[0].to(args.device)
                transcr = data[1]
                style_images = data[3].to(args.device)
                text_features = tokenizer(
                    transcr,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                    max_length=200,
                ).to(args.device)

                reshaped_images = style_images.reshape(-1, 3, 64, 256)

                if style_extractor is not None:
                    style_features = style_extractor(reshaped_images).to(args.device)
                else:
                    style_features = None

                if args.latent == True:
                    x = torch.randn(
                        (
                            images.size(0),
                            4,
                            self.img_size[0] // 8,
                            self.img_size[1] // 8,
                        )
                    ).to(args.device)

                else:
                    x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(
                        args.device
                    )

                # scheduler
                noise_scheduler.set_timesteps(50)
                for time in noise_scheduler.timesteps:

                    t_item = time.item()
                    t = (torch.ones(images.size(0)) * t_item).long().to(args.device)

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

        with torch.no_grad():
            style_images = None
            text_features = [x_text] * n
            # print('text features', text_features.shape)
            text_features = tokenizer(
                text_features,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=40,
            ).to(args.device)
            if args.img_feat == True:
                # pick random image according to specific style
                with open("utils/writers_dict_train.json", "r") as f:

                    wr_dict = json.load(f)
                reverse_wr_dict = {v: k for k, v in wr_dict.items()}

                # key = reverse_wr_dict[value]
                with open("./utils/splits_words/iam_train_val.txt", "r") as f:
                    # with open('./utils/splits_words/iam_test.txt', 'r') as f:
                    train_data = f.readlines()
                    train_data = [i.strip().split(",") for i in train_data]
                    for label in labels:
                        # print('label', label)
                        label_index = label.item()

                        matching_lines = [
                            line
                            for line in train_data
                            if line[1] == reverse_wr_dict[label_index]
                            and len(line[2]) > 3
                        ]

                        # pick the first 5 from matching lines
                        if len(matching_lines) >= 5:
                            five_styles = random.sample(matching_lines, 5)
                        else:
                            matching_lines = [
                                line
                                for line in train_data
                                if line[1] == reverse_wr_dict[label_index]
                            ]
                            # print('matching lines', matching_lines)
                            five_styles = matching_lines_style[:5]
                            five_styles = [matching_lines[0]] * 5
                            # five_styles = random.sample(matching_lines, 5)
                        # print("five_styles", five_styles)
                        # five_styles = random.sample(matching_lines, 5)

                        cor_image_random = random.sample(matching_lines, 1)

                        interpol = False
                        if interpol == True:
                            label2 = random.randint(0, 339)  # random label
                            matching_lines2 = [
                                line
                                for line in train_data
                                if line[1] == reverse_wr_dict[label2]
                                and len(line[2]) > 3
                            ]
                            five_styles = random.sample(matching_lines2, 5)
                        # print('five_styles', five_styles)
                        fheight, fwidth = 64, 256
                        root_path = "./iam_data/words"
                        cor_im = False
                        if cor_im == True:
                            cor_image = Image.open(
                                os.path.join(root_path, cor_image_random[0][0])
                            ).convert(
                                "RGB"
                            )  # ['a05/a05-089/a05-089-00-05.png', '000', 'debate']
                            (cor_image_width, cor_image_height) = cor_image.size
                            cor_image = cor_image.resize(
                                (int(cor_image_width * 64 / cor_image_height), 64)
                            )
                            (cor_image_width, cor_image_height) = cor_image.size

                            if cor_image_width < 256:
                                outImg = ImageOps.pad(
                                    cor_image, size=(256, 64), color="white"
                                )  # , centering=(0,0)) uncommment to pad right
                                cor_image = outImg

                            else:
                                # reduce image until width is smaller than 256
                                while cor_image_width > 256:
                                    cor_image = image_resize_PIL(
                                        cor_image, width=cor_image_width - 20
                                    )
                                    (cor_image_width, cor_image_height) = cor_image.size
                                cor_image = centered_PIL(
                                    cor_image, (64, 256), border_value=255.0
                                )

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
                            # print('file_path', file_path)

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
                                outImg = ImageOps.pad(
                                    img_s, size=(256, 64), color="white"
                                )  # , centering=(0,0)) uncommment to pad right
                                img_s = outImg

                            else:
                                # reduce image until width is smaller than 256
                                while img_width > 256:
                                    img_s = image_resize_PIL(
                                        img_s, width=img_width - 20
                                    )
                                    (img_width, img_height) = img_s.size
                                img_s = centered_PIL(
                                    img_s, (64, 256), border_value=255.0
                                )
                            # make grid of all 5 images
                            # img_s = img_s.convert('L')
                            transform_tensor = transforms.ToTensor()
                            grid_im = transform_tensor(img_s)
                            grid_imgs += [grid_im]

                            img_tens = transform(img_s).to(args.device)  # .unsqueeze(0)
                            st_imgs += [img_tens]
                            # style_features = style_extractor(style_images).to(args.device)
                            # img_tensor = img_tensor.to(args.device)
                        s_imgs = torch.stack(st_imgs).to(args.device)
                        style_images = (
                            torch.cat((style_images, s_imgs))
                            if style_images is not None
                            else s_imgs
                        )

                        grid_imgs = torch.stack(grid_imgs).to(args.device)

                        style_images = style_images.to(args.device)

                    # save style images
                    style_images = style_images.reshape(-1, 3, 64, 256)
                    style_features = style_extractor(style_images).to(args.device)
                    # style_features = torch.stack(style_featur, dim=0) #We get [320, 5, 2048]
                    # print('style features', style_features.shape)
                    # style_features = style_features.reshape(n, -1).to(args.device)
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
