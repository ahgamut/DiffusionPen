import torch
import argparse

#
from utils.generation import (
    setup_logging,
    render_paragraph,
)
from utils.arghandle import add_common_args, file_check
from utils.gen_cli import read_words
from utils.model_setup import load_models


def main():
    """Main function"""
    parser = argparse.ArgumentParser("interp-paragraph")
    parser.add_argument("--writer-1", type=int, default=1)
    parser.add_argument("--writer-2", type=int, default=3)
    parser.add_argument("-i", "--text-file", type=file_check, default="./sample.txt")
    parser.add_argument("-o", "--output", type=str, default="./output.png")
    parser.add_argument(
        "--max-line-width", default=900, type=int, help="max line width"
    )
    add_common_args(parser)
    parser.set_defaults(interpolation=True)

    args = parser.parse_args()
    assert args.mix_rate is not None
    print(__file__, "with torch", torch.__version__)

    # create save directories
    setup_logging(args)
    torch.cuda.empty_cache()

    m = load_models(args)

    words = read_words(args.text_file)
    max_line_width = args.max_line_width

    paragraph_image = render_paragraph(
        words, args, m, max_line_width=max_line_width, interp=True
    )
    paragraph_image.save(args.output)


if __name__ == "__main__":
    main()
