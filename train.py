import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
import torchvision
from torch import optim
from tqdm import tqdm
import copy
import argparse
import json
from diffusers import AutoencoderKL, DDIMScheduler
from torch.nn import DataParallel
from transformers import CanineModel, CanineTokenizer
from torchvision import transforms

#
from models import UNetModel, ImageEncoder, EMA, Diffusion, AvgMeter
from utils.word_dataset import MergedWordDataset
from utils.auxiliary_functions import *
from utils.generation import save_image_grid, setup_logging
from utils.arghandle import add_common_args

torch.cuda.empty_cache()
OUTPUT_MAX_LEN = 95  # + 2  # <GO>+groundtruth+<END>
IMG_WIDTH = 256
IMG_HEIGHT = 64

c_classes = (
    "_!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "
)
cdict = {c: i for i, c in enumerate(c_classes)}
icdict = {i: c for i, c in enumerate(c_classes)}


def labelDictionary():
    labels = list(c_classes)
    letter2index = {label: n for n, label in enumerate(labels)}
    # create json object from dictionary if you want to save writer ids
    json_dict_l = json.dumps(letter2index)
    l = open("utils/letter2index.json", "w")
    l.write(json_dict_l)
    l.close()
    index2letter = {v: k for k, v in letter2index.items()}
    json_dict_i = json.dumps(index2letter)
    l = open("utils/index2letter.json", "w")
    l.write(json_dict_i)
    l.close()
    return len(labels), letter2index, index2letter


char_classes, letter2index, index2letter = labelDictionary()
tok = False
if not tok:
    tokens = {"PAD_TOKEN": 52}
else:
    tokens = {"GO_TOKEN": 52, "END_TOKEN": 53, "PAD_TOKEN": 54}
num_tokens = len(tokens.keys())
print("num_tokens", num_tokens)


print("num of character classes", char_classes)
vocab_size = char_classes + num_tokens


def build_MergedDataset(args, transform):
    """The merged IAM+CVL+CSAFE memmap split (built by utils/build_multidataset.py)
    -- the only training dataset in this codebase.

    ``style_classes`` is the merged writer count W (read off the loaded split), so
    the style bank / placer / model must be sized to the same W. ``--data-dir``
    points straight at the split directory produced by the builder."""
    train_data = MergedWordDataset(
        args.data_dir, transforms=transform, args=args,
    )
    style_classes = train_data.wclasses
    print("merged writers (style_classes):", style_classes)

    test_size = args.batch_size
    rest = len(train_data) - test_size
    test_data, _ = random_split(
        train_data, [test_size, rest], generator=torch.Generator().manual_seed(42)
    )
    return train_data, test_data, style_classes


def train(
    diffusion,
    model,
    ema,
    ema_model,
    vae,
    optimizer,
    mse_loss,
    loader,
    test_loader,
    num_classes,
    style_extractor,
    vocab_size,
    noise_scheduler,
    transforms,
    args,
    tokenizer=None,
    text_encoder=None,
    lr_scheduler=None,
):
    model.train()
    loss_meter = AvgMeter()
    print("Training started....")

    # When latents are precomputed, data[0] is already the VAE latent mean and
    # the per-step vae.encode is skipped (see build_latent_cache).
    use_latent_cache = getattr(loader.dataset, "latent_cache", None) is not None

    for epoch in range(args.epochs):
        print("Epoch:", epoch)
        pbar = tqdm(loader, desc="\n::")
        for i, data in enumerate(pbar):
            images = data[0].to(args.device)
            transcr = data[1]
            s_id = data[2].to(args.device)
            style_images = data[3].to(args.device)

            text_features = tokenizer(
                transcr,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=40,
            ).to(args.device)

            if style_images.dim() == 3:
                # Cached per-image style features (bs, 5, feat) -> (bs*5, feat).
                style_features = style_images.reshape(-1, style_images.size(-1))
            elif style_extractor is not None:
                reshaped_images = style_images.reshape(-1, 3, 64, 256)
                style_features = style_extractor(reshaped_images)

            else:
                style_features = None

            if args.latent :
                if not use_latent_cache:
                    images = vae.module.encode(
                        images.to(torch.float32)
                    ).latent_dist.sample()
                images = images * 0.18215

            noise = torch.randn(images.shape).to(images.device)
            # Sample a random timestep for each image
            num_train_timesteps = diffusion.noise_steps

            timesteps = torch.randint(
                0, num_train_timesteps, (images.shape[0],), device=images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude
            # at each timestep (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
            x_t = noisy_images
            t = timesteps

            predicted_noise = model(
                x_t,
                timesteps=t,
                context=text_features,
                y=s_id,
                style_extractor=style_features,
            )

            loss = mse_loss(noise, predicted_noise)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            ema.step_ema(ema_model, model)

            count = images.size(0)
            loss_meter.update(loss.item(), count)
            pbar.set_postfix(MSE=loss_meter.avg)

            if lr_scheduler is not None:
                lr_scheduler.step()

        print("train MSE:", repr(loss_meter))

        if epoch % 10 == 0:
            if args.sample_preview:
                labels = torch.arange(16).long().to(args.device)
                n = len(labels)

                words = ["text", "sample", "images"]
                for x_text in words:
                    ema_sampled_images = diffusion.sampling(
                        ema_model,
                        vae,
                        n=n,
                        x_text=x_text,
                        labels=labels,
                        args=args,
                        style_extractor=style_extractor,
                        noise_scheduler=noise_scheduler,
                        transform=transforms,
                        tokenizer=tokenizer,
                        text_encoder=text_encoder,
                    )

                    epoch_n = epoch
                    sampled_ema = save_image_grid(
                        ema_sampled_images,
                        os.path.join(
                            args.save_path, "images", f"{epoch_n:04d}_{x_text}_ema.jpg"
                        ),
                        args,
                    )

            torch.save(
                model.state_dict(), os.path.join(args.save_path, "models", "ckpt.pt")
            )
            torch.save(
                ema_model.state_dict(),
                os.path.join(args.save_path, "models", "ema_ckpt.pt"),
            )
            torch.save(
                optimizer.state_dict(),
                os.path.join(args.save_path, "models", "optim.pt"),
            )


def load_style_weights(model, device, style_path):
    print(
        "Number of model parameters: {}".format(
            sum([p.data.nelement() for p in model.parameters()])
        )
    )
    style_state_dict = torch.load(style_path, map_location=device, weights_only=True)
    model_dict = model.state_dict()
    sub_dict = dict()
    for k, v in style_state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            sub_dict[k] = v
        else:
            print("skipping style weights for: ", k)
    model_dict.update(sub_dict)
    model.load_state_dict(model_dict)
    print("Pretrained style model loaded")


def build_style_cache(dataset, extractor, args):
    """Precompute and cache the frozen style-CNN features for every image.

    The style extractor is frozen + eval, so its output for a given image is
    deterministic. We compute it once for all dataset images, cache [N, feat]
    to disk (keyed on the style checkpoint + dataset identity so a changed
    checkpoint rebuilds), and attach it to the dataset. __getitem__ then gathers
    the 5 sampled style vectors by index instead of running the CNN on 5 crops
    every step -- which also slashes the per-batch host->device transfer.
    """
    os.makedirs(args.style_cache_path, exist_ok=True)
    key = "{}_{}_{}".format(
        os.path.splitext(os.path.basename(args.style_path.rstrip("/")))[0],
        getattr(dataset, "setname", "data"),
        len(dataset.data),
    )
    cache_file = os.path.join(args.style_cache_path, key + ".pt")

    if os.path.isfile(cache_file):
        feats = torch.load(cache_file, map_location="cpu", weights_only=True)
        print("Loaded style feature cache:", cache_file, tuple(feats.shape))
    else:
        print("Building style feature cache ->", cache_file)
        extractor.eval()
        n = len(dataset.data)
        bs = args.batch_size
        chunks = []
        # _img() fetches the crop from either the memmap array or data[i][0].
        get_img = getattr(dataset, "_img", lambda i: dataset.data[i][0])
        with torch.no_grad():
            for start in tqdm(range(0, n, bs), desc="style-cache"):
                imgs = [
                    dataset.transforms(get_img(i))
                    for i in range(start, min(start + bs, n))
                ]
                batch = torch.stack(imgs).to(args.device)
                chunks.append(extractor(batch).detach().cpu())
        feats = torch.cat(chunks, dim=0)
        torch.save(feats, cache_file)
        print("Saved style feature cache:", cache_file, tuple(feats.shape))

    dataset.style_cache = feats
    return feats


def build_latent_cache(dataset, vae, args):
    """Precompute and cache the frozen VAE latent for every image.

    The VAE is frozen and the crops are fixed (transforms are deterministic:
    ToTensor + Normalize, no augmentation), so vae.encode(img) is the same on
    every epoch -- yet the training loop re-runs it every step. We compute the
    latent distribution *mean* once for all images and cache [N, 4, 8, 32] to
    disk (keyed on the VAE checkpoint + dataset identity so a changed VAE
    rebuilds); __getitem__ then returns the cached latent as the image and the
    loop skips vae.encode.

    The mean (not a .sample()) is cached: for a frozen VAE the posterior std is
    tiny, so dropping the per-step sampling jitter is a negligible, standard
    trade for removing the encode from the hot loop. The 0.18215 scaling stays
    in the loop so both the cached and non-cached paths scale identically.

    Scale-safe by construction (same philosophy as the images.npy memmap):
    latents stream into an ``open_memmap`` .npy row-by-row (only one batch in
    RAM at build time, no [N,...] concat) as float32, and are read back with
    ``mmap_mode="r"`` so the OS pages them on demand and forked DataLoader
    workers share the mapping -- RAM stays O(1) in N, not O(N). The cache lives
    inside the split dir next to images.npy (keyed on the VAE checkpoint, so a
    changed VAE rebuilds), and travels with the dataset.
    """
    vae_name = os.path.splitext(os.path.basename(args.stable_dif_path.rstrip("/")))[0]
    cache_file = os.path.join(dataset.data_dir, "latents_{}.npy".format(vae_name))
    n = len(dataset.data)

    if os.path.isfile(cache_file):
        feats = np.load(cache_file, mmap_mode="r")
        if feats.ndim == 4 and feats.shape[0] == n:
            print("Loaded VAE latent cache:", cache_file, tuple(feats.shape))
            dataset.latent_cache = feats
            return feats
        print("stale VAE latent cache (shape {} vs N={}); rebuilding".format(
            tuple(feats.shape), n))
        del feats

    print("Building VAE latent cache ->", cache_file)
    bs = args.batch_size
    get_img = getattr(dataset, "_img", lambda i: dataset.data[i][0])
    arr = None
    with torch.no_grad():
        for start in tqdm(range(0, n, bs), desc="latent-cache"):
            imgs = [
                dataset.transforms(get_img(i))
                for i in range(start, min(start + bs, n))
            ]
            batch = torch.stack(imgs).to(args.device).to(torch.float32)
            mean = vae.module.encode(batch).latent_dist.mean.detach().cpu().numpy()
            if arr is None:
                # allocate now that the latent shape is known
                arr = np.lib.format.open_memmap(
                    cache_file, mode="w+", dtype=np.float32, shape=(n,) + mean.shape[1:]
                )
            arr[start:start + mean.shape[0]] = mean.astype(np.float32)
    arr.flush()
    del arr
    feats = np.load(cache_file, mmap_mode="r")
    print("Saved VAE latent cache:", cache_file, tuple(feats.shape))
    dataset.latent_cache = feats
    return feats


def build_preview_style_bank(dataset, extractor, num_writers, args):
    """Per-writer mean style vector [num_writers, feat] from the merged dataset,
    set as diffusion.style_bank so the preview conditions on merged writers."""
    by_writer = dataset.writer_to_indices
    feats_by_writer = {}
    if dataset.style_cache is not None:
        cache = dataset.style_cache
        feat_dim = cache.size(-1)
        for wid, idxs in by_writer.items():
            feats_by_writer[wid] = cache[idxs].mean(dim=0)
    else:
        extractor.eval()
        feat_dim = None
        with torch.no_grad():
            for wid, idxs in tqdm(by_writer.items(), desc="preview-bank"):
                batch = torch.stack(
                    [dataset.transforms(dataset._img(i)) for i in idxs[:5]]
                ).to(args.device)
                vec = extractor(batch).mean(dim=0).detach().cpu()
                feats_by_writer[wid] = vec
                feat_dim = vec.numel()

    bank = torch.zeros(num_writers, feat_dim)
    for wid, vec in feats_by_writer.items():
        if 0 <= wid < num_writers:
            bank[wid] = vec.to(bank.dtype)
    print("built preview style bank:", tuple(bank.shape))
    return bank


def main():
    """Main function"""
    parser = argparse.ArgumentParser("diffusionpen-train")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=320)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--level", type=str, default="word", help="word, line")
    parser.add_argument(
        "--data-dir", type=str,
        default="./saved_iam_data/combined_word_train",
        help="path to the merged dataset split directory built by "
        "utils/build_multidataset.py",
    )
    parser.add_argument("--style-cache", dest="style_cache", action="store_true")
    parser.add_argument("--no-style-cache", dest="style_cache", action="store_false")
    parser.add_argument("--style-cache-path", type=str, default="./saved_style_cache")
    parser.add_argument("--latent-cache", dest="latent_cache", action="store_true")
    parser.add_argument("--no-latent-cache", dest="latent_cache", action="store_false")
    parser.add_argument("--sample-preview", dest="sample_preview", action="store_true")
    parser.add_argument(
        "--no-sample-preview", dest="sample_preview", action="store_false"
    )
    parser.set_defaults(style_cache=True, latent_cache=True, sample_preview=True)
    add_common_args(parser)
    args = parser.parse_args()

    print("torch version", torch.__version__)

    # create save directories
    setup_logging(args)

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

    # Single training dataset: the merged memmap split built by
    # utils/build_multidataset.py (see MergedWordDataset).
    print("loading merged (IAM+CVL+CSAFE) memmap split")
    train_data, test_data, style_classes = build_MergedDataset(args, transform)

    # Worker-dependent flags only make sense with >0 workers.
    loader_kwargs = dict(pin_memory=True)
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        **loader_kwargs,
    )
    character_classes = get_default_character_classes()

    ######################### MODEL #######################################
    vocab_size = len(character_classes)
    print("Vocab size: ", vocab_size)

    if args.dataparallel :
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

    if args.load_check :
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

    if args.latent :
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
    if args.style_name == "mobilenetv2_100":
        feature_extractor = ImageEncoder(
            model_name="mobilenetv2_100", num_classes=0, pretrained=True, trainable=True
        )
    elif args.style_name == "resnet18":
        feature_extractor = ImageEncoder(
            model_name="resnet18", num_classes=0, pretrained=True, trainable=True
        )
    else:
        raise ValueError(f"unable to load style model {style_name}!")
    load_style_weights(feature_extractor, args.device, args.style_path)
    feature_extractor = DataParallel(feature_extractor, device_ids=device_ids)
    feature_extractor = feature_extractor.to(args.device)
    feature_extractor.requires_grad_(False)
    feature_extractor.eval()

    # Precompute frozen style features so the per-step 5-crop CNN pass is skipped.
    if args.style_cache:
        build_style_cache(train_data, feature_extractor, args)

    # Precompute frozen VAE latents so the per-step vae.encode is skipped.
    if args.latent and args.latent_cache:
        build_latent_cache(train_data, vae, args)

    if args.sample_preview:
        diffusion.style_bank = build_preview_style_bank(
            train_data, feature_extractor, style_classes, args
        )

    train(
        diffusion,
        unet,
        ema,
        ema_model,
        vae,
        optimizer,
        mse_loss,
        train_loader,
        test_loader,
        style_classes,
        feature_extractor,
        vocab_size,
        ddim,
        transform,
        args,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        lr_scheduler=lr_scheduler,
    )


if __name__ == "__main__":
    main()
