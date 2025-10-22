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

    parser.set_defaults(
        color=True,
        latent=True,
        img_feat=True,
        interpolation=False,
        dataparallel=False,
        load_check=False,
    )
