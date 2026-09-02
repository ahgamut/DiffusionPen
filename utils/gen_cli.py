"""Shared CLI scaffold for the generation scripts.

Every generation entry point repeats the same prologue -- parse args, announce
the torch version, create the logging/output dirs, clear the CUDA cache, and
load the model bundle -- and reads a whitespace-separated prompt file. These
helpers collapse that boilerplate; scripts still declare their own argparse
options and keep any custom setup (e.g. IAM_TempLoader.check_preload) around the
call.
"""

import torch

from utils.arghandle import file_check
from utils.generation import setup_logging
from utils.model_setup import load_models


# Shared generation CLI options. Several scripts declared these identically
# (only the -o default varied), so they are factored here; each script adds the
# ones it needs. Flag spellings + types stay in one place.
def add_writer_id_arg(parser, default=12):
    parser.add_argument("-w", "--writer-id", type=int, default=default)


def add_text_file_arg(parser, default="./sample.txt"):
    parser.add_argument("-i", "--text-file", type=file_check, default=default)


def add_output_arg(parser, default="./output.png"):
    parser.add_argument("-o", "--output", type=str, default=default)


def add_max_line_width_arg(parser):
    parser.add_argument(
        "--max-line-width", default=900, type=int, help="max line width"
    )


def init_generation(parser, script):
    """Parse args, run the standard prologue, and load the model bundle.

    ``script`` is the caller's ``__file__`` (used only for the startup banner).
    Returns ``(args, models)`` where ``models`` is the ``load_models`` dict the
    builders consume.
    """
    args = parser.parse_args()
    print(script, "with torch", torch.__version__)
    setup_logging(args)
    torch.cuda.empty_cache()
    return args, load_models(args)


def read_words(path):
    """Read a whitespace-separated prompt file into a list of words."""
    return open(path).read().strip().split(" ")
