import os
import torch
import numpy as np
from PIL import Image
import argparse

#
from utils.generation import (
    setup_logging,
    build_fake_interp_1,
)
from utils.arghandle import add_common_args
from utils.model_setup import load_models


def img_concat(imgs):
    w = max(x.width for x in imgs)
    h = sum(x.height for x in imgs)
    dst = Image.new("RGB", (w, h))
    ch = 0
    for img in imgs:
        dst.paste(img, (0, ch))
        ch += img.height
    return dst


def main():
    """Main function"""
    parser = argparse.ArgumentParser("diffusion-bulk-interp")
    parser.add_argument("--tag", default="i1", help="tag")
    parser.add_argument("--writer-1", type=int, default=1)
    parser.add_argument("--writer-2", type=int, default=3)
    parser.add_argument("--sampling-word", type=str, default="hello")
    parser.add_argument("-o", "--output", type=str, default="./outputs/")
    add_common_args(parser)
    parser.set_defaults(interpolation=False)

    args = parser.parse_args()
    print(__file__, "with torch", torch.__version__)

    # create save directories
    setup_logging(args)
    torch.cuda.empty_cache()

    m = load_models(args)
    diffusion = m["diffusion"]
    ema_model = m["ema_model"]
    vae = m["vae"]
    ddim = m["ddim"]
    feature_extractor = m["feature_extractor"]
    transform = m["transform"]
    tokenizer = m["tokenizer"]
    text_encoder = m["text_encoder"]

    w = 0.1
    weights = np.arange(0, 1 + w, w)
    for writer_1 in range(1, 5):
        for writer_2 in range(writer_1 + 1, 5):
            imgs = []
            for weight in weights:
                args.writer_1 = writer_1
                args.writer_2 = writer_2
                args.mix_rate = weight

                im = build_fake_interp_1(
                    args=args,
                    diffusion=diffusion,
                    ema_model=ema_model,
                    vae=vae,
                    feature_extractor=feature_extractor,
                    ddim=ddim,
                    transform=transform,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                )
                imgs.append(im)
            dst = img_concat(imgs)
            dst.save(
                os.path.join(
                    args.output,
                    f"{args.tag}-{args.sampling_word}-{writer_1}-{writer_2}.png",
                )
            )


if __name__ == "__main__":
    main()
