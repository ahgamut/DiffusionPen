import pandas as pd
import numpy as np
import random
import json
import sys
import os
import argparse


def make_cs_writers(multi, remain, required, padding=50):
    actuals = set(random.sample(list(multi), required))
    padded = set(random.sample(list(remain), padding))
    return actuals, padded


def make_cs_RHS(df, actuals, padded):
    rhs_actual = []
    for wid in actuals:
        subdf = df[df["wid"] == wid]
        rsu = random.randint(0, len(subdf) - 1)
        # print(subdf)
        act = subdf.iloc[rsu].name
        rhs_actual.append(act)
        # print("act is ", act)
    #
    rhs_padded = []
    for wid in padded:
        subdf = df[df["wid"] == wid]
        rsu = random.randint(0, len(subdf) - 1)
        # print(subdf)
        pad = subdf.iloc[rsu].name
        rhs_padded.append(pad)
        # print("pad is ", pad)

    act_df = df[df.apply(lambda x: x.name in rhs_actual, axis=1)].copy()
    act_df["actual_writer"] = 1
    pad_df = df[df.apply(lambda x: x.name in rhs_padded, axis=1)].copy()
    pad_df["actual_writer"] = 0

    # rhs_df = pd.concat([act_df, pad_df])
    # print(rhs_df)
    return act_df, pad_df


def subsample_data(df, ratio=0.2):
    if ratio > 0:
        subind = random.sample(range(0, len(df)), int(ratio * len(df)))
        subres = df.iloc[subind, :].copy()
        return subres
    else:
        return df


def remake_fakes(df):
    rparts = []
    for fake_type in [
        "fake-niceplace",
        "fake-traintext",
        "fake-traintext-12-img",
        "fake-traintext-13-img",
        "fake-traintext-14-img",
        "fake-traintext-rsp-img",
        "fake-traintext-12-font",
        "fake-traintext-14-font",
        "fake-traintext-16-font",
        "fake-traintext-rsp-font",
        # "fake-difftext1",
        "fake-difftext1-12-img",
        "fake-difftext1-13-img",
        "fake-difftext1-14-img",
        "fake-difftext1-rsp-img",
        "fake-difftext1-12-font",
        "fake-difftext1-14-font",
        "fake-difftext1-16-font",
        "fake-difftext1-rsp-font",
        # "fake-difftext2",
        "fake-difftext2-12-img",
        "fake-difftext2-13-img",
        "fake-difftext2-14-img",
        "fake-difftext2-rsp-img",
        "fake-difftext2-12-font",
        "fake-difftext2-14-font",
        "fake-difftext2-16-font",
        "fake-difftext2-rsp-font",
    ]:
        dupe = df.copy()
        dupe["file2_type"] = fake_type
        rparts.append(dupe)

    res_df = pd.concat(rparts)
    return res_df


def fill_renames(df, prefix):
    df["target_name"] = [f"{prefix}-{x:04d}" for x in range(1, len(df) + 1)]


def make_matches_real(df, actuals, act_df, mm=3):
    res = []
    # for each writer, pick 3 matching pairs at random
    for i, arow in act_df.iterrows():
        wid = arow["wid"]
        subdf = df[df["wid"] == wid]
        rsu = random.sample(range(0, len(subdf)), mm)
        lhs_df = subdf.iloc[rsu, :]
        # print(lhs_df)

        for j, lrow in lhs_df.iterrows():
            row_dict = {
                "file1_path": arow["imgname"],
                "file1_type": "real",
                "file1_wid": arow["wid"],
                "file2_path": lrow["imgname"],
                "file2_wid": lrow["wid"],
                "file2_type": "real",
                "same_wid": 1,
            }
            res.append(row_dict)

    res_df = pd.DataFrame(res)
    return res_df


def make_nonmatches_real(df, actuals, act_df, subsamp=0):
    res = []
    mm = 1
    for i, arow in act_df.iterrows():
        for j, brow in act_df.iterrows():
            if j <= i:
                continue
            wid2 = brow["wid"]
            subdf = df[df["wid"] == wid2]
            rsu = random.sample(range(0, len(subdf)), mm)
            lhs_df = subdf.iloc[rsu, :]
            # print(lhs_df)

            for j, lrow in lhs_df.iterrows():
                row_dict = {
                    "file1_path": arow["imgname"],
                    "file1_type": "real",
                    "file1_wid": arow["wid"],
                    "file2_path": lrow["imgname"],
                    "file2_wid": lrow["wid"],
                    "file2_type": "real",
                    "same_wid": 0,
                }
                res.append(row_dict)

    res_df = pd.DataFrame(res)
    res_df = subsample_data(res_df, subsamp)

    return res_df


def make_matches_fake(df, actuals, act_df, mm, subsamp=0):
    res = []
    for i, arow in act_df.iterrows():
        wid = arow["wid"]
        subdf = df[df["wid"] == wid]
        rsu = random.sample(range(0, len(subdf)), mm)
        lhs_df = subdf.iloc[rsu, :]
        # print(lhs_df)

        for j, lrow in lhs_df.iterrows():
            row_dict = {
                "file1_path": arow["imgname"],
                "file1_type": "real",
                "file1_wid": arow["wid"],
                "file2_path": lrow["imgname"],
                "file2_wid": lrow["wid"],
                "file2_type": "fake-niceplace",
                "same_wid": 1,
            }
            res.append(row_dict)

    res_df = pd.DataFrame(res)
    res_df = subsample_data(res_df, subsamp)
    # after subsampling, pick fakes
    rfull = remake_fakes(res_df)
    return rfull


def make_nonmatches_fake(df, actuals, act_df, subsamp=0):
    res = []
    mm = 1
    subo_n = 3
    for i, arow in act_df.iterrows():
        wid = arow["wid"]
        others = list(actuals - set([wid]))
        sub_others = set(random.sample(others, subo_n))
        for j, brow in act_df.iterrows():
            if j <= i:
                continue
            wid2 = brow["wid"]
            subdf = df[df["wid"] == wid2]
            rsu = random.sample(range(0, len(subdf)), mm)
            lhs_df = subdf.iloc[rsu, :]
            # print(lhs_df)

            for j, lrow in lhs_df.iterrows():
                row_dict = {
                    "file1_path": arow["imgname"],
                    "file1_type": "real",
                    "file1_wid": arow["wid"],
                    "file2_path": lrow["imgname"],
                    "file2_wid": lrow["wid"],
                    "file2_type": "fake-niceplace",
                    "same_wid": 0,
                }
                res.append(row_dict)

    res_df = pd.DataFrame(res)
    res_df = subsample_data(res_df, subsamp)
    # after subsampling, pick fakes
    rfull = remake_fakes(res_df)
    return rfull


def make_interps(df, actuals, act_df, n_anchors, subsamp=0):
    res = []
    #
    alphas = list(np.arange(0.0, 1.1, 0.1))
    anchor_wids = random.sample(list(actuals), n_anchors)

    for i in range(len(anchor_wids)):
        wid1 = anchor_wids[i]
        anchor1 = act_df[act_df["wid"] == wid1].iloc[0, :]
        for j in range(i + 1, len(anchor_wids)):
            wid2 = anchor_wids[j]
            anchor2 = act_df[act_df["wid"] == wid2].iloc[0, :]

            for a in alphas:
                row_dict1 = {
                    "file1_path": anchor1["imgname"],
                    "file1_type": "real",
                    "file1_wid": anchor1["wid"],
                    "file2_path": f"{wid1}-{wid2}-{a:.2f}",
                    "file2_type": f"sametext-{a:.2f}",
                    "file2_wid": "interp",
                    "same_wid": f"{a:.2f}",
                }
                row_dict2 = {
                    "file1_path": anchor1["imgname"],
                    "file1_type": "real",
                    "file1_wid": anchor1["wid"],
                    "file2_path": f"{wid1}-{wid2}-{a:.2f}",
                    "file2_type": f"difftext-{a:.2f}",
                    "file2_wid": "interp",
                    "same_wid": f"{a:.2f}",
                }
                res.append(row_dict1)
                res.append(row_dict2)

    res_df = pd.DataFrame(res)
    res_df = subsample_data(res_df, 0)
    return res_df


def get_multi_remain(df, mm=3):
    train_wr = json.load(open("./utils/writers_dict_train_iam.json", "r"))
    train_keys = set(f'"{x}"' for x in train_wr.keys())
    #
    multi_wids = set()
    remain_wids = set()
    for wid, gdf in df.groupby("wid"):
        if len(gdf) > mm and wid != '"000"':
            multi_wids.add(wid)
        else:
            remain_wids.add(wid)
        # print(wid, len(gdf))
    # print(smulti_wids)
    return multi_wids, remain_wids, train_keys


def runner(
    input_csv, outdir, required, padding, use_train_only, mm, n_anchors, subsample
):
    #
    df = pd.read_csv(input_csv)
    print("originally had", len(df), "XMLs")
    df = df[df["parsed"] == 1]
    print("parsed", len(df), "XMLs")
    #
    multi_wids, remain_wids, train_keys = get_multi_remain(df, mm)
    print(len(multi_wids), "entries have >=", mm, "docs/writer")
    smulti_wids = multi_wids & train_keys
    print(len(smulti_wids), "entries have >=", mm, "docs/writer and in train set")

    if use_train_only:
        multi_wids = multi_wids & train_keys
    actuals, padded = make_cs_writers(
        multi_wids, remain_wids, required=required, padding=padding
    )

    act_df, pad_df = make_cs_RHS(df, actuals, padded)
    # bind these two, save as closed_set csv
    closed_set_df = pd.concat([act_df, pad_df])

    matches_real = make_matches_real(df, actuals, act_df, mm=mm)
    nonmatches_real = make_nonmatches_real(df, actuals, act_df, subsample)
    matches_fake = make_matches_fake(df, actuals, act_df, mm=mm, subsamp=subsample)
    nonmatches_fake = make_nonmatches_fake(df, actuals, act_df, subsamp=0.1)
    interps = make_interps(df, actuals, act_df, n_anchors, subsample)

    df_mapping = {
        "clref": closed_set_df,
        "qmreal": matches_real,
        "qnreal": nonmatches_real,
        "qmfake": matches_fake,
        "qnfake": nonmatches_fake,
        "qinterp": interps,
    }

    # add target filenames to avoid dupe complaints
    for k, v in df_mapping.items():
        fill_renames(v, k)
        if k in ["clref", "qmreal", "qmfake"]:
            print(k, "has", len(v), "entries")
            v.to_csv(os.path.join(outdir, f"samp-{k}.csv"), header=True, index=False)
        else:
            print(k, "has", len(v), "entries, picking only 20")
            subv = v.iloc[:20, :]
            subv.to_csv(os.path.join(outdir, f"samp-{k}.csv"), header=True, index=False)



def main():
    parser = argparse.ArgumentParser("sampling-scheme")
    parser.add_argument(
        "-i", "--input-csv", default="./saved_iam_data/IAM_valids.csv", help="input csv"
    )
    parser.add_argument("-o", "--outdir", default="./saved_iam_data", help="output dir")
    parser.add_argument(
        "-r",
        "--required",
        dest="required",
        default=10,
        type=int,
        help="number of true writers in closed set",
    )
    parser.add_argument(
        "-p",
        "--padding",
        default=10,
        type=int,
        help="number of writers to add to pad closed set",
    )
    parser.add_argument(
        "--min-reps",
        default=3,
        type=int,
        help="minimum number of repeats to be considered a true writer",
    )
    parser.add_argument(
        "--subsample", default=0, type=float, help="provide fraction to subsample"
    )
    parser.add_argument(
        "--use-train-only",
        dest="use_train_only",
        action="store_true",
        help="if true only uses writers in the training set",
    )
    parser.add_argument(
        "-a",
        "--n-anchors",
        default=5,
        type=int,
        help="number of writer anchors for interpolation",
    )

    parser.set_defaults(use_train_only=False)
    d = parser.parse_args()
    assert (d.subsample >= 0) & (d.subsample <= 1)
    assert d.n_anchors < d.required

    runner(
        input_csv=d.input_csv,
        outdir=d.outdir,
        required=d.required,
        padding=d.padding,
        mm=d.min_reps,
        subsample=d.subsample,
        use_train_only=d.use_train_only,
        n_anchors=d.n_anchors,
    )


if __name__ == "__main__":
    main()
