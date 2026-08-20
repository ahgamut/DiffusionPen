import argparse

#
from utils.generation import build_fake_image_1
from utils.arghandle import add_common_args
from utils.gen_cli import init_generation


def main():
    parser = argparse.ArgumentParser("diffusionpen-singleword")
    add_common_args(parser)
    parser.add_argument("-w", "--writer-id", type=int, default=12)
    parser.add_argument("--sampling-word", type=str, default="hello")
    parser.add_argument("-o", "--output", default="./output.png", help="output")

    args, m = init_generation(parser, __file__)

    word = args.sampling_word
    writer_id = args.writer_id  # index for style class

    image = build_fake_image_1(
        word,
        writer_id,
        args,
        m,
    )
    image.save(args.output)


if __name__ == "__main__":
    main()
