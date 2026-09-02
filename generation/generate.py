import argparse

#
from utils.generation import build_fake_image_1
from utils.arghandle import add_common_args
from utils.gen_cli import init_generation, add_writer_id_arg, add_output_arg


def main():
    parser = argparse.ArgumentParser("diffusionpen-singleword")
    add_common_args(parser)
    add_writer_id_arg(parser)
    parser.add_argument("--sampling-word", type=str, default="hello")
    add_output_arg(parser)

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
