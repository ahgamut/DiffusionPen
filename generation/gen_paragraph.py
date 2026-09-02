import argparse

#
from utils.generation import (
    build_fake_image_N,
    add_rescale_padding,
    build_paragraph_image,
    place_words_learned,
    upsample_words,
)
from utils.arghandle import add_common_args
from utils.gen_cli import (
    init_generation,
    read_words,
    add_writer_id_arg,
    add_text_file_arg,
    add_output_arg,
    add_max_line_width_arg,
)


def main():
    parser = argparse.ArgumentParser("gen-paragraph")
    add_writer_id_arg(parser)
    add_text_file_arg(parser)
    add_output_arg(parser)
    add_max_line_width_arg(parser)
    parser.add_argument(
        "--placement",
        choices=["heuristic", "learned"],
        default="heuristic",
        help="word layout: font-metric heuristic or learned WordPlacer",
    )
    parser.add_argument(
        "--placer-path",
        type=str,
        default=None,
        help="checkpoint for the learned WordPlacer (required for --placement learned)",
    )
    parser.add_argument(
        "--upsample",
        action="store_true",
        help="upscale each word crop 2x before layout (learned WordUpsampler if "
        "--upsampler-path is given, else Lanczos)",
    )
    parser.add_argument(
        "--upsampler-path",
        type=str,
        default=None,
        help="checkpoint for the learned WordUpsampler (optional; Lanczos fallback)",
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
    )

    # optional super-resolution of each word crop before layout
    up_scale = 1
    if args.upsample:
        up_scale = 2
        fakes = upsample_words(fakes, args, m, scale=up_scale)

    if args.placement == "learned":
        if m["placer"] is None:
            raise RuntimeError(
                "--placement learned requires --placer-path to a valid checkpoint"
            )
        paragraph_image = place_words_learned(
            words,
            fakes,
            writer_id=s,
            args=args,
            models=m,
            max_line_width=max_line_width * up_scale,
            ref_height=64 * up_scale,
        )
    else:
        # Scale and pad each word
        scaled_padded_words = add_rescale_padding(
            words,
            fakes,
            max_word_length_width=max_word_length_width,
            longest_word_length=longest_word_length,
        )

        # combine to create paragraph
        paragraph_image = build_paragraph_image(
            scaled_padded_words, max_line_width=max_line_width
        )
    paragraph_image.save(args.output)


if __name__ == "__main__":
    main()
