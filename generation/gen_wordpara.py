import os
import argparse

#
from utils.generation import build_fake_image_N
from utils.arghandle import add_common_args, file_check
from utils.gen_cli import init_generation, read_words


def main():
    parser = argparse.ArgumentParser("gen-wordpara")
    parser.add_argument("-w", "--writer-id", type=int, default=12)
    parser.add_argument("-i", "--text-file", type=file_check, default="./sample.txt")
    parser.add_argument("-o", "--output", type=str, default="./outputs/")
    parser.add_argument(
        "--max-line-width", default=900, type=int, help="max line width"
    )
    add_common_args(parser)

    args, m = init_generation(parser, __file__)

    words = read_words(args.text_file)
    max_line_width = args.max_line_width
    max_word_length_width = 0
    longest_word_length = max(len(word) for word in words)
    s = args.writer_id

    # build fake images
    fakes, max_word_length_width = build_fake_image_N(
        words,
        s=s,
        args=args,
        models=m,
        longest_word_length=longest_word_length,
        max_word_length_width=max_word_length_width,
        crop_whitespace=False,
    )

    for word, img in zip(words, fakes):
        img.save(os.path.join(args.output, f"gen_{s}_{word}.png"))


if __name__ == "__main__":
    main()
