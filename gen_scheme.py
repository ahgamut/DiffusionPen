import argparse
import os
import pandas as pd


def make_closedset(fname, targdir):
    pass


def process_csv(fname, targdir):
    print("processing", fname)
    if "clref" in fname:
        make_closedset(fname, targdir)
        return


def main():
    parser = argparse.ArgumentParser("generate-scheme")
    parser.add_argument(
        "-d",
        "--config-dir",
        default="./saved_iam_data",
        help="file containing config CSVs",
    )
    parser.add_argument(
        "-o", "--output-dir", default="./saved_iam_data", help="output dir"
    )

    d = parser.parse_args()

    pieces = ["clref", "qmreal", "qnreal", "qmfake", "qnfake", "qinterp"]
    for p in pieces:
        fname = os.path.join(d.config_dir, f"samp-{p}.csv")
        targdir = os.path.join(d.output_dir, p)
        os.makedirs(targdir, exist_ok=True)
        process_csv(fname, targdir)


if __name__ == "__main__":
    main()
