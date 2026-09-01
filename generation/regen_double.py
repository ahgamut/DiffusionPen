import sys
import traceback
import random
import os
import argparse

#
from utils.generation import render_paragraph
from utils.arghandle import add_common_args, file_check
from utils.gen_cli import init_generation, read_words


def main():
    parser = argparse.ArgumentParser("regen-double")
    parser.add_argument("-n", "--num-samples", type=int, default=5)
    parser.add_argument("-i", "--text-file", type=file_check, default="./sample.txt")
    parser.add_argument("-o", "--output", type=str, default="./outputs")
    parser.add_argument(
        "--max-line-width", default=900, type=int, help="max line width"
    )
    add_common_args(parser)

    args, m = init_generation(parser, __file__)

    words = read_words(args.text_file)

    for i in range(args.num_samples):
        s = random.randint(0, 338)
        try:
            max_line_width = args.max_line_width
            # generate the same prompt twice with the same writer
            gen_1 = render_paragraph(
                words, args, m, max_line_width=max_line_width, s=s
            )
            gen_2 = render_paragraph(
                words, args, m, max_line_width=max_line_width, s=s
            )

            rid = "%04x" % random.randint(0, 1000)
            gen_1.save(os.path.join(args.output, f"gen_{s}_{rid}_1.png"))
            gen_2.save(os.path.join(args.output, f"gen_{s}_{rid}_2.png"))
        except Exception as e:
            print(e)
            tb = traceback.format_tb(sys.exc_info()[2])
            print("".join(tb))


if __name__ == "__main__":
    main()
