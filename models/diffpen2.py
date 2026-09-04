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

    def get_style_coll(self, label_index, args):
        # Tile the writer's mean vector to [5, 1280] so the UNet reshape->mean
        # returns it. The bank is the only style source at inference.
        if self.style_bank is None:
            raise RuntimeError(
                "inference requires a style bank; pass --style-bank / --style-bank-path"
            )
        n_writers = self.style_bank.shape[0]
        if not (0 <= label_index < n_writers):
            raise IndexError(
                "writer id {} out of range for style bank [0, {}); the bank's "
                "writer-id space does not match this model -- rebuild it from the "
                "same split with utils/build_style_bank.py".format(
                    label_index, n_writers
                )
            )
        vec = self.style_bank[label_index].to(args.device)
        if not torch.any(vec):
            raise RuntimeError(
                "style bank row for writer id {} is all zeros (unpopulated writer). "
                "The bank is stale or was built from a different split -- every "
                "writer id would collapse to the same output. Rebuild it with "
                "utils/build_style_bank.py (matching --data-dir/--style-name/"
                "--style-path to the trained checkpoint).".format(label_index)
            )
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

    def _dog_guidance(self, eps_p, eps_n, t, g_s=0.0, u_T=700, tau=0.0, total_T=None,
                      skip=None):
        """DOG guidance (arXiv:2508.17017): eps^ = eps_p + g(t)*(eps_p - eps*),
        eps* = eps_n - proj_{eps_p}(eps_n), g(t) triangular in t (peak u_T).
        Per-sample; g_s=0 is a no-op; tau<=0 skips the negative norm-clip. ``skip``
        is an optional per-sample bool mask; masked samples pass eps_p through."""
        if g_s == 0:
            return eps_p
        if total_T is None:
            total_T = self.noise_steps
        b = eps_p.shape[0]
        flat_p = eps_p.reshape(b, -1)
        flat_n = eps_n.reshape(b, -1)
        if tau and tau > 0:  # optional per-sample norm-clip (Eq. 8)
            scale = torch.clamp(tau / (flat_n.norm(dim=1, keepdim=True) + 1e-12), max=1.0)
            flat_n = flat_n * scale
        denom = (flat_p * flat_p).sum(dim=1, keepdim=True) + 1e-12
        coef = (flat_n * flat_p).sum(dim=1, keepdim=True) / denom
        eps_star = (flat_n - coef * flat_p).reshape_as(eps_p)
        gamma = (t / u_T) if t <= u_T else (1.0 - (t - u_T) / (total_T - u_T))
        g = g_s * float(gamma)
        guided = eps_p + g * (eps_p - eps_star)
        if skip is not None:
            m = torch.as_tensor(skip, device=eps_p.device, dtype=torch.bool)
            m = m.view(-1, *([1] * (eps_p.dim() - 1)))
            guided = torch.where(m, eps_p, guided)
        return guided

    @staticmethod
    def _masked_noise(shape, lam, p, device):
        """DOG negative rep r~ = lam * eta * N(0,I), eta ~ Bernoulli(p) (Eq. 4)."""
        mask = torch.bernoulli(torch.full(shape, float(p), device=device))
        return lam * mask * torch.randn(shape, device=device)

    def _dog_negative_params(self, args, n, model, model_params):
        """Negative-condition kwargs: swap style/text rep for masked noise per
        --dog-neg (both|style|text); other kwargs unchanged. Fresh each call."""
        device = args.device
        p = getattr(args, "dog_keep_prob", 0.75)
        which = getattr(args, "dog_neg", "both")
        neg = dict(model_params)
        sf = model_params.get("style_extractor")
        if sf is not None and which in ("both", "style"):
            feat = sf.shape[-1]
            lam_s = getattr(args, "dog_lambda_s", 1000.0)
            rs = self._masked_noise((n, feat), lam_s, p, device)
            # tile to [n*5, feat]; UNet reshape(b,5,-1).mean returns rs
            neg["style_extractor"] = (
                rs.unsqueeze(1).expand(n, 5, feat).reshape(n * 5, feat)
            )
        ctx = model_params.get("context")
        if ctx is not None and which in ("both", "text"):
            lam_t = getattr(args, "dog_lambda_t", 1000.0)
            seq = ctx["input_ids"].shape[1]
            core = getattr(model, "module", model)  # unwrap DataParallel
            cont_dim = getattr(core, "cont_dim", 768)
            # [n,seq,cont_dim] fed directly (forward bypasses the encoder)
            neg["context"] = self._masked_noise((n, seq, cont_dim), lam_t, p, device)
        return neg

    def update_schedule_x(
        self,
        args,
        n,
        x,
        noise_scheduler,
        model,
        model_params,
        dog_skip=None,
    ):
        g_s = getattr(args, "dog_gs", 0.0)
        u_T = getattr(args, "dog_ut", 700)
        tau = getattr(args, "dog_tau", 0.0)
        # skip the negative pass entirely when every sample opts out (all short)
        dog_on = g_s > 0 and (dog_skip is None or not all(dog_skip))
        noise_scheduler.set_timesteps(50)
        for time in noise_scheduler.timesteps:
            t_item = time.item()
            t = (torch.ones(n) * t_item).long().to(args.device)
            eps_p = model(
                x,
                timesteps=t,
                **model_params,
            )
            if dog_on:  # DOG: negative pass + orthogonal guidance
                neg_params = self._dog_negative_params(args, n, model, model_params)
                eps_n = model(x, timesteps=t, **neg_params)
                noisy_residual = self._dog_guidance(
                    eps_p, eps_n, t_item, g_s=g_s, u_T=u_T, tau=tau, skip=dog_skip
                )
            else:
                noisy_residual = eps_p
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
            # skip DOG on short words (weak positive prediction -> blow-up)
            min_chars = getattr(args, "dog_min_chars", 1)
            dog_skip = (
                [len(str(w)) < min_chars for w in x_text]
                if isinstance(x_text, (list, tuple)) and min_chars > 1
                else None
            )
            x = self.update_schedule_x(
                args, n, x, noise_scheduler, model, model_params, dog_skip=dog_skip
            )

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
