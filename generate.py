import torch
import argparse

#
from utils.generation import (
    setup_logging,
    build_fake_image,
)
from utils.arghandle import add_common_args
from utils.model_setup import load_models


def main():
    parser = argparse.ArgumentParser("diffusionpen-singleword")
    add_common_args(parser)
    parser.add_argument("-w", "--writer-id", type=int, default=12)
    parser.add_argument("--sampling-word", type=str, default="hello")
    parser.add_argument("-o", "--output", default="./output.png", help="output")

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

    word = args.sampling_word
    writer_id = args.writer_id  # index for style class

    image = build_fake_image(
        word,
        writer_id,
        args,
        diffusion,
        ema_model,
        vae,
        feature_extractor,
        ddim,
        transform,
        tokenizer,
        text_encoder,
    )
    image.save(args.output)


if __name__ == "__main__":
    main()
