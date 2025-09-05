#


def add_common_args(parser):
    parser.add_argument(
        "--model-name",
        type=str,
        default="diffusionpen",
        help="(deprecated)",
    )
    parser.add_argument("--dataset", default="iam", help="iam, gnhk, cvl")
    parser.add_argument("--img-size", type=int, default=(64, 256))
    # UNET parameters
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--emb-dim", type=int, default=320)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-res-blocks", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--color", type=bool, default=True)
    parser.add_argument("--latent", type=bool, default=True)
    parser.add_argument("--img-feat", type=bool, default=True)
    parser.add_argument("--interpolation", type=bool, default=False)
    parser.add_argument("--dataparallel", type=bool, default=False)
    parser.add_argument("--load-check", type=bool, default=False)
    parser.add_argument("--mix-rate", type=float, default=None)
    # file paths
    parser.add_argument(
        "--save-path", type=str, default="./diffusionpen_iam_model_path"
    )
    parser.add_argument(
        "--style-path", type=str, default="./style_models/iam_style_diffusionpen.pth"
    )
    parser.add_argument(
        "--stable-dif-path", type=str, default="./stable-diffusion-v1-5"
    )
