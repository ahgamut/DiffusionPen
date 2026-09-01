import sys
import traceback
import argparse

#
from utils.generation import render_paragraph
from utils.arghandle import add_common_args, file_check, range_check
from utils.gen_cli import init_generation, read_words


def main():
    """Main function"""
    parser = argparse.ArgumentParser("diffusion-paragraph-bulk")
    parser.add_argument("-w", "--writer-range", type=range_check, default=(1, 1))
    parser.add_argument(
        "--max-line-width", default=900, type=int, help="max line width"
    )
    parser.add_argument("-i", "--text-file", type=file_check, default="./sample.txt")
    parser.add_argument("-o", "--output", type=str, default="./output.png")
    add_common_args(parser)

    args, m = init_generation(parser, __file__)

    words = read_words(args.text_file)
    max_line_width = args.max_line_width

    output_template = args.output.replace(".png", "-{s}.png")
    writer_range = args.writer_range
    for s in range(writer_range[0], writer_range[1] + 1):
        try:
            paragraph_image = render_paragraph(
                words, args, m, max_line_width=max_line_width, s=s
            )
            paragraph_image.save(output_template.format(s=s))
        except Exception as e:
            print("failed for", s)
            print(e)
            tb = traceback.format_tb(sys.exc_info()[2])
            print("".join(tb))


if __name__ == "__main__":
    main()
