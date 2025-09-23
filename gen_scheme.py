import argparse
import os
import pandas as pd
from PIL import Image, ImageOps

#
from utils.subprompt import Word, Prompt


def save_threshed(img, fname):
    # thresh separately? use Otsu?
    # thresh here?
    # tmp = ImageOps.autocontrast(img)
    img.save(fname)


def resave_real(xmlname, imgname, targname):
    prompt = Prompt(xmlname)
    img = Image.open(imgname).convert("RGB")
    crop = prompt.get_cropped(img)
    save_threshed(crop, targname)


def make_closedset(fname, targdir):
    df = pd.read_csv(fname)
    for ind, row in df.iterrows():
        wid = row["wid"].replace('"', "")
        xmlname = os.path.join("./iam_data/xml/", row["xmlname"])
        imgname = os.path.join("./iam_data/forms", row["imgname"])
        targname = os.path.join(targdir, row["target_name"]) + ".png"
        resave_real(xmlname, imgname, targname)


def resave_fake(xmlname, imgname, targname, faketype):
    if "niceplace" in faketype:
        print("should regenerate", imgname, "place nicely and save")
    elif "traintext" in faketype:
        print("should regenerate", imgname, "place however and save")
    elif "difftext1" in faketype:
        print("should generate LL using wid from", imgname, "and save")
    elif "difftext2" in faketype:
        print("should generate WOZ using wid from", imgname, "and save")


def resave_interp(xmlname, imgname, targname, widinfo, interp):
    wid1, wid2, alpha = widinfo.split("-")
    wid1 = wid1.replace('"', "")
    wid2 = wid2.replace('"', "")
    alpha = float(alpha)
    if "sametext" in interp:
        print(
            "should interpolate between",
            (wid1, wid2),
            "at",
            alpha,
            "use same text and save to",
            targname,
        )
    else:
        print(
            "should interpolate between",
            (wid1, wid2),
            "at",
            alpha,
            "use different text and save to",
            targname,
        )


def process_csv(fname, targdir):
    print("processing", fname)
    if "clref" in fname:
        make_closedset(fname, targdir)
        return
    #
    df = pd.read_csv(fname)
    for ind, row in df.iterrows():
        wid = row["file2_wid"].replace('"', "")
        proc_tp = row["file2_type"]
        targname = os.path.join(targdir, row["target_name"]) + ".png"

        if proc_tp == "real":
            img_basename = row["file2_path"]
            imgname = os.path.join("./iam_data/forms", img_basename)
            xmlname = os.path.join("./iam_data/xml", img_basename.replace("png", "xml"))
            resave_real(xmlname, imgname, targname)

        elif proc_tp.startswith("fake-"):
            img_basename = row["file2_path"]
            imgname = os.path.join("./iam_data/forms", img_basename)
            xmlname = os.path.join("./iam_data/xml", img_basename.replace("png", "xml"))
            resave_fake(xmlname, imgname, targname, proc_tp)

        else:
            anchor_basename = row["file1_path"]
            imgname = os.path.join("./iam_data/forms", anchor_basename)
            xmlname = os.path.join(
                "./iam_data/xml", anchor_basename.replace("png", "xml")
            )
            widinfo = row["file2_path"]
            resave_interp(xmlname, imgname, targname, widinfo, proc_tp)


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
