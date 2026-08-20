import argparse

#
from utils.generation import build_fake_image_N
from utils.relcharsize import build_placed_paragraph
from utils.arghandle import add_common_args, file_check
from utils.gen_cli import init_generation, read_words


def main():
    parser = argparse.ArgumentParser("gen-rcs")
    parser.add_argument("-w", "--writer-id", type=int, default=12)
    parser.add_argument("-i", "--text-file", type=file_check, default="./sample.txt")
    parser.add_argument("-o", "--output", type=str, default="./output.png")
    parser.add_argument(
        "--max-line-width", default=900, type=int, help="max line width"
    )
    parser.add_argument("--font-size", default=16, type=int, help="font size")
    parser.add_argument("--dpi", default=300, help="DPI")
    parser.add_argument(
        "--image-aspect",
        action="store_true",
        dest="use_aspect",
        help="use aspect ratio from image",
    )
    parser.add_argument(
        "--font-aspect",
        action="store_false",
        dest="use_aspect",
        help="use aspect ratio from font",
    )

    add_common_args(parser)
    parser.set_defaults(use_aspect=True)

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
        crop_whitespace=True,
    )

    # combine to create paragraph
    paragraph_image = build_placed_paragraph(
        words,
        fakes,
        max_line_width=max_line_width,
        font_size=args.font_size,
        dpi=args.dpi,
        use_aspect=args.use_aspect,
    )
    paragraph_image.save(args.output)


if __name__ == "__main__":
    main()
