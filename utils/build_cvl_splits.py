import argparse
import glob
import os
import random


def dir_exists(x):
    a = os.path.exists(x)
    b = os.path.isdir(x)
    return a and b


def check_cvl_root_folder(fname):
    if not dir_exists(fname):
        raise RuntimeError(f"cvl root folder {fname} is not valid!")

    # the root folder of CVL data should
    # have the following subfolders:
    s1 = ["trainset", "testset"]
    s2 = ["pages", "lines", "words", "xml"]

    for s1i in s1:
        for s2j in s2:
            subdir = os.path.join(fname, s1i, s2j)
            if not dir_exists(subdir):
                raise RuntimeError(
                    f"cvl root folder {fname} should contain {s1i}/{s2j}"
                )

    return fname


def split_check(num):
    num = int(num)
    if num < 0 or n > 100:
        raise RuntimeError("{num} is not a valid percentage!")
    return num


def check_results_folder(fname):
    if not dir_exists(fname):
        raise RuntimeError(f"results folder {fname} is not valid!")
    return fname


def wpath_is_ascii(x):
    return x.isascii()


def get_writer_id(x):
    bpath = os.path.splitext(os.path.basename(x))[0]
    wid = bpath.split("-")[0]
    return wid


def get_actual_word(x):
    bpath = os.path.splitext(os.path.basename(x))[0]
    words = bpath.split("-")[4:]
    if len(words) > 1:
        word = "-".join(words)
    else:
        word = words[0]
    return word


def get_wpath_coll(root_folder, only_ascii):
    train_words = os.path.join(root_folder, "trainset", "words", "*", "*.tif")
    test_words = os.path.join(root_folder, "testset", "words", "*", "*.tif")
    files0 = glob.glob(train_words) + glob.glob(test_words)
    if only_ascii:
        files1 = filter(wpath_is_ascii, files0)
    else:
        files1 = files0

    files = files1
    result = dict()
    for f in files:
        wid = get_writer_id(f)
        if wid not in result:
            result[wid] = []
        result[wid].append(f)
    return result


def get_splits(wpath_coll, train_split):
    train_perc = train_split / 100.0
    test_perc = 1.0 - train_perc
    val_perc = 0.2

    train_list = []
    val_list = []
    test_list = []

    for k, v in wpath_coll.items():
        n = len(v)
        n_test = max(1, int(test_perc * n))
        #
        n_tv = n - n_test
        n_val = max(1, int(val_perc * n_tv))
        #
        n_train = n_tv - n_val
        #
        wpaths = v
        random.shuffle(wpaths)
        test_list += wpaths[:n_test]
        train_list += wpaths[n_test : (n_test + n_train)]
        val_list += wpaths[(n_test + n_train) :]

    return train_list, val_list, test_list


def wpaths_to_file(root_folder, fname, wpath_list):
    with open(fname, "w") as f:
        for wpath in wpath_list:
            wid = get_writer_id(wpath)
            word = get_actual_word(wpath)
            assert root_folder in wpath
            relpath = wpath.replace(root_folder, "")
            f.write(f"{relpath},{wid},{word}\n")


def main():
    parser = argparse.ArgumentParser(
        prog="build-cvl-data",
        description="build training and test splits for CVL database",
    )
    parser.add_argument(
        "-i",
        "--cvl-folder",
        default="./cvl_data",
        type=check_cvl_root_folder,
        help="path to root of CVL data",
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        type=check_results_folder,
        default="./utils/splits_words",
        help="folder to store results",
    )
    parser.add_argument(
        "-s",
        "--train-split",
        default=80,
        type=split_check,
        help="training split percentage",
    )
    parser.add_argument(
        "--only-ascii",
        dest="only_ascii",
        action="store_true",
        help="(default) allow only ascii chars",
    )
    parser.add_argument(
        "--use-all-chars",
        dest="only_ascii",
        action="store_false",
        help="allow all chars",
    )
    parser.set_defaults(only_ascii=True)
    args = parser.parse_args()
    #
    wpath_coll = get_wpath_coll(args.cvl_folder, args.only_ascii)
    train_list, test_list, val_list = get_splits(wpath_coll, args.train_split)
    train_val_list = train_list + val_list
    #
    wpaths_to_file(args.cvl_folder, os.path.join(args.output_folder, "cvl_training.txt"), train_list)
    wpaths_to_file(args.cvl_folder, os.path.join(args.output_folder, "cvl_val.txt"), val_list)
    wpaths_to_file(
        args.cvl_folder, os.path.join(args.output_folder, "cvl_train_val.txt"), train_val_list
    )
    wpaths_to_file(args.cvl_folder, os.path.join(args.output_folder, "cvl_test.txt"), test_list)


if __name__ == "__main__":
    main()
