import random
import numpy as np
import argparse

#
from utils.generation import (
    render_paragraph,
    stack_images,
)
from utils.arghandle import add_common_args
from utils.gen_cli import (
    init_generation,
    read_words,
    add_text_file_arg,
    add_output_arg,
    add_max_line_width_arg,
)


def main():
    """Main function"""
    parser = argparse.ArgumentParser("interp-viz")
    add_text_file_arg(parser)
    add_output_arg(parser)
    add_max_line_width_arg(parser)
    add_common_args(parser)
    parser.set_defaults(interpolation=True)

    args, m = init_generation(parser, __file__)

    words = read_words(args.text_file)
    max_line_width = args.max_line_width

    writers = [random.randint(0, 338) for x in range(4)]
    base_wt = 0.25
    wt_pieces = np.arange(1.00, -0.001, -base_wt)
    big_images = []

    for i in range(len(writers) - 1):
        args.writer_1 = writers[i]
        args.writer_2 = writers[i + 1]
        for wt in wt_pieces:
            if (wt == 1) and i != 0:
                continue
            args.mix_rate = wt
            paragraph_image = render_paragraph(
                words, args, m, max_line_width=max_line_width, interp=True
            )
            big_images.append(paragraph_image)

    res_image = stack_images(big_images, margin=5)
    res_image.save(args.output)


if __name__ == "__main__":
    main()
