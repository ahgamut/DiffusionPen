import sys
import traceback
import random
import os
import argparse

#
from utils.generation import render_paragraph
from utils.arghandle import add_common_args
from utils.gen_cli import (
    init_generation,
    read_words,
    add_text_file_arg,
    add_output_arg,
    add_max_line_width_arg,
)


def main():
    parser = argparse.ArgumentParser("regen-double")
    parser.add_argument("-n", "--num-samples", type=int, default=5)
    add_text_file_arg(parser)
    add_output_arg(parser, default="./outputs")
    add_max_line_width_arg(parser)
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
