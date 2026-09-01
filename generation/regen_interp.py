import sys
import traceback
import random
import os
import numpy as np
import argparse

#
from utils.generation import render_paragraph
from utils.arghandle import add_common_args, file_check
from utils.gen_cli import init_generation, read_words


def main():
    parser = argparse.ArgumentParser("regen-interp")
    parser.add_argument("-n", "--num-samples", type=int, default=5)
    parser.add_argument("-i", "--text-file", type=file_check, default="./sample.txt")
    parser.add_argument("-o", "--output", type=str, default="./outputs")
    parser.add_argument(
        "--max-line-width", default=900, type=int, help="max line width"
    )
    add_common_args(parser)

    args, m = init_generation(parser, __file__)

    words = read_words(args.text_file)
    max_line_width = args.max_line_width
    w = 0.1
    weights = np.arange(0, 1 + w, w)

    for i in range(args.num_samples):
        rid = "%04x" % random.randint(0, 1000)
        s1 = random.randint(0, 338)
        s2 = random.randint(0, 338)
        while s2 == s1:
            s2 = random.randint(0, 338)
        args.writer_1 = s1
        args.writer_2 = s2

        try:
            # generate with s1
            gen_1 = render_paragraph(
                words, args, m, max_line_width=max_line_width, s=s1
            )
            gen_1.save(os.path.join(args.output, f"intgen_{s1}_{rid}_1.png"))

            # generate with s2
            gen_2 = render_paragraph(
                words, args, m, max_line_width=max_line_width, s=s2
            )
            gen_2.save(os.path.join(args.output, f"intgen_{s2}_{rid}_1.png"))

            for weight in weights:
                # generate with interpolated style
                args.mix_rate = weight
                gen_int = render_paragraph(
                    words, args, m, max_line_width=max_line_width, interp=True
                )
                gen_int.save(
                    os.path.join(
                        args.output, f"intgen_{s1}_{s2}_{weight:.1f}_{rid}.png"
                    )
                )
        except Exception as e:
            print(e)
            tb = traceback.format_tb(sys.exc_info()[2])
            print("".join(tb))


if __name__ == "__main__":
    main()
