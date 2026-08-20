import torch


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
        # Optional precomputed per-writer style bank [num_writers, 1280]
        # (utils/build_style_bank.py). When set, get_style_coll looks up the
        # writer's mean vector instead of reading 5 random crops + running the
        # CNN -- faster, reproducible, lower-variance. See stage-3 Part A.
        self.style_bank = None

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def get_style_coll(self, label_index, args):
        # Bank lookup: tile the writer's mean vector to [5, 1280] so the UNet's
        # reshape(b,5,1280)->mean returns exactly that mean. The precomputed
        # style bank is the only style source at inference (the raw-IAM CNN path
        # was removed); callers must load one via --style-bank / --style-bank-path.
        if self.style_bank is None:
            raise RuntimeError(
                "inference requires a style bank; pass --style-bank / --style-bank-path"
            )
        vec = self.style_bank[label_index].to(args.device)
        s_feat = vec.unsqueeze(0).expand(5, -1).contiguous()
        return {"images": None, "features": s_feat}

    def get_text_embed(self, x_text, tokenizer, max_length=40):
        n = 0
        if isinstance(x_text, list):
            n = len(x_text)
        else:
            n = 1
        text_features = tokenizer(
            x_text,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=40,
        )
        return n, text_features

    def get_initial_x(self, args, n, noise_scheduler, cor_im=False):
        if args.latent:
            x = torch.randn((n, 4, self.img_size[0] // 8, self.img_size[1] // 8)).to(
                args.device
            )
            if cor_im:
                x_noise = torch.randn(cor_images.shape).to(args.device)
                timesteps = torch.full(
                    (cor_images.shape[0],),
                    999,
                    device=args.device,
                    dtype=torch.long,
                )
                noisy_images = noise_scheduler.add_noise(cor_images, x_noise, timesteps)
                x = noisy_images
        else:
            x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(args.device)
        return x

    def update_schedule_x(
        self,
        args,
        n,
        x,
        noise_scheduler,
        model,
        model_params,
    ):
        noise_scheduler.set_timesteps(50)
        for time in noise_scheduler.timesteps:
            t_item = time.item()
            t = (torch.ones(n) * t_item).long().to(args.device)
            noisy_residual = model(
                x,
                timesteps=t,
                **model_params,
            )
            prev_noisy_sample = noise_scheduler.step(
                noisy_residual, time, x
            ).prev_sample
            x = prev_noisy_sample
        return x

    def post_process_x(self, args, x, vae):
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
            text_features = [x_text] * n
            text_features = tokenizer(
                text_features,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=40,
            ).to(args.device)

            style_colls = []

            if args.img_feat:
                for label in labels:
                    style_colls.append(
                        self.get_style_coll(label.item(), args)
                    )
                style_images = style_colls[0]["images"]
                # [n*5, feat] so the UNet reshape(b,5,-1) uses each image's writer
                style_features = torch.cat(
                    [sc["features"] for sc in style_colls], dim=0
                )
            else:
                style_images = None
                style_features = None

            #
            x = self.get_initial_x(args, n, noise_scheduler, cor_im=False)

            # scheduler
            model_params = dict(
                context=text_features,
                y=labels,
                original_images=style_images,
                mix_rate=mix_rate,
                style_extractor=style_features,
            )
            x = self.update_schedule_x(args, n, x, noise_scheduler, model, model_params)

        model.train()
        return self.post_process_x(args, x, vae)

    def interp_1(
        self,
        model,
        vae,
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
        assert len(labels) == 2
        n = 1

        if mix_rate is None:
            mix_rate = args.mix_rate
        print("mix_rate", mix_rate)

        with torch.no_grad():
            text_features = x_text
            text_features = tokenizer(
                text_features,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=40,
            ).to(args.device)

            style_colls = []

            if args.img_feat:
                for label in labels:
                    style_colls.append(
                        self.get_style_coll(label.item(), args)
                    )

                style_images = style_colls[0]["images"]
                style_features = style_colls[0]["features"] * mix_rate + style_colls[1][
                    "features"
                ] * (1 - mix_rate)
            else:
                style_images = None
                style_features = None

            x = self.get_initial_x(args, n, noise_scheduler, cor_im=False)

            model_params = dict(
                context=text_features,
                original_images=style_images,
                style_extractor=style_features,
            )
            x = self.update_schedule_x(args, n, x, noise_scheduler, model, model_params)

        model.train()
        return self.post_process_x(args, x, vae)

    def sampling_bulk(
        self,
        model,
        vae,
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
        assert args.img_feat
        assert len(labels) == 1

        with torch.no_grad():
            n, text_features = self.get_text_embed(x_text, tokenizer)
            text_features = text_features.to(args.device)

            style_colls = []
            for i in range(n):
                style_colls.append(self.get_style_coll(labels[0].item(), args))
            style_features = torch.stack([x["features"] for x in style_colls])

            #
            x = self.get_initial_x(args, n, noise_scheduler, cor_im=False)

            # scheduler
            model_params = dict(
                context=text_features,
                y=labels,
                original_images=None,
                mix_rate=mix_rate,
                style_extractor=style_features,
            )
            x = self.update_schedule_x(args, n, x, noise_scheduler, model, model_params)

        model.train()
        return self.post_process_x(args, x, vae)

    def interp_bulk(
        self,
        model,
        vae,
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
        assert args.img_feat
        assert len(labels) == 2
        n = 1

        if mix_rate is None:
            mix_rate = args.mix_rate

        with torch.no_grad():
            n, text_features = self.get_text_embed(x_text, tokenizer)
            text_features = text_features.to(args.device)

            sc0 = []
            sc1 = []
            for i in range(n):
                sc0.append(self.get_style_coll(labels[0].item(), args))
                sc1.append(self.get_style_coll(labels[1].item(), args))
            sf0 = torch.stack([x["features"] for x in sc0])
            sf1 = torch.stack([x["features"] for x in sc1])
            style_features = sf0 * mix_rate + sf1 * (1 - mix_rate)

            #
            x = self.get_initial_x(args, n, noise_scheduler, cor_im=False)

            # scheduler
            model_params = dict(
                context=text_features,
                original_images=None,
                style_extractor=style_features,
            )
            x = self.update_schedule_x(args, n, x, noise_scheduler, model, model_params)

        model.train()
        return self.post_process_x(args, x, vae)
