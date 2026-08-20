#
import os


def file_check(fname):
    if os.path.isfile(fname):
        return fname
    raise RuntimeError(f"{fname} is not a file")


def range_check(x):
    l, u = x.split("-")
    l = int(l)
    u = int(u)
    if l < 0 or u < 0 or l > u:
        raise RuntimeError(f"invalid range: {x}")
    return (l, u)


def add_common_args(parser):
    parser.add_argument(
        "--model-name",
        type=str,
        default="diffusionpen",
        help="(deprecated)",
    )
    parser.add_argument(
        "--dataset", default="iam",
        help="IAM-space tag for build_style_bank / generation; training ignores it "
        "(train.py always uses the merged MergedWordDataset)",
    )
    parser.add_argument(
        "--style-classes", type=int, default=0,
        help="override the writer-class count at inference (0 = auto-detect from "
        "the UNet checkpoint's label_emb; set W for a merged-trained model)",
    )
    parser.add_argument("--img-size", type=int, default=(64, 256))
    # UNET parameters
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--emb-dim", type=int, default=320)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-res-blocks", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mix-rate", type=float, default=None)
    parser.add_argument(
        "--max-word-chars",
        type=int,
        default=12,
        help="split words longer than this into fixed-canvas chunks that are "
        "generated separately and butt-joined (0 disables chunking)",
    )
    # file paths
    parser.add_argument(
        "--save-path", type=str, default="./diffusionpen_iam_model_path"
    )
    parser.add_argument(
        "--style-path", type=str, default="./style_models/iam_style_diffusionpen.pth"
    )
    parser.add_argument("--style-name", type=str, default="mobilenetv2_100")
    parser.add_argument(
        "--stable-dif-path", type=str, default="./stable-diffusion-v1-5"
    )
    # boolean arguments
    parser.add_argument("--color", dest="color", action="store_true")
    parser.add_argument("--no-color", dest="color", action="store_false")
    parser.add_argument("--latent", dest="latent", action="store_true")
    parser.add_argument("--no-latent", dest="latent", action="store_false")
    parser.add_argument("--img-feat", dest="img_feat", action="store_true")
    parser.add_argument("--no-img-feat", dest="img_feat", action="store_false")
    parser.add_argument("--interpolation", dest="interpolation", action="store_true")
    parser.add_argument(
        "--no-interpolation", dest="interpolation", action="store_false"
    )
    parser.add_argument("--dataparallel", dest="dataparallel", action="store_true")
    parser.add_argument("--no-dataparallel", dest="dataparallel", action="store_false")
    parser.add_argument("--load-check", dest="load_check", action="store_true")
    parser.add_argument("--no-load-check", dest="load_check", action="store_false")
    # Precomputed per-writer style bank (stage-3 Part A). Used when enabled AND
    # the file exists; otherwise generation falls back to the 5-crop CNN path.
    parser.add_argument("--style-bank", dest="style_bank", action="store_true")
    parser.add_argument("--no-style-bank", dest="style_bank", action="store_false")
    parser.add_argument(
        "--style-bank-path", type=str, default="./saved_iam_data/style_bank.pt"
    )

    parser.set_defaults(
        color=True,
        latent=True,
        img_feat=True,
        interpolation=False,
        dataparallel=False,
        load_check=False,
        style_bank=True,
    )
