#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
report_echonet.py
Usage:
  python report_echonet.py --data_dir <DATA_DIR> --seg_dir output/segmentation --vid_dir output/video --n_curves 3
"""

import os
import re
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def find_latest_csv(root, filename):
    cands = []
    for r, d, f in os.walk(root):
        if filename in f:
            cands.append(os.path.join(r, filename))
    if not cands:
        return None
    return sorted(cands, key=os.path.getmtime)[-1]

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def load_test_df(test_pred_csv, data_dir):
    pred = pd.read_csv(test_pred_csv)
    cols = {c.lower(): c for c in pred.columns}
    # 預測欄位名稱兼容：Pred / Prediction / EF_Pred / EF
    yhat_col = cols.get("pred") or cols.get("prediction") or cols.get("ef_pred") or cols.get("ef")
    # 檔名欄位兼容：FileName / Filename / File / StudyName
    fname_col = cols.get("filename") or cols.get("file") or cols.get("studyname") or cols.get("filename_x")
    if fname_col is None:
        # 嘗試抓第一個像檔名的欄位
        for c in pred.columns:
            if "file" in c.lower() or "name" in c.lower():
                fname_col = c; break
    if yhat_col is None or fname_col is None:
        raise RuntimeError(f"Cannot find prediction/filename columns in {test_pred_csv}")

    # 若 test_predictions.csv 沒有真值 EF，和 FileList.csv 合併
    has_true = any(c.lower() == "ef" for c in pred.columns if c.lower() != yhat_col.lower())
    if not has_true:
        fl = pd.read_csv(os.path.join(data_dir, "FileList.csv"))
        pred["__key__"] = pred[fname_col].astype(str).str.replace(r"\.avi$","",regex=True)
        fl["__key__"]   = fl["FileName"].astype(str).str.replace(r"\.avi$","",regex=True)
        df = pred.merge(fl[["__key__","EF","Split"]], on="__key__", how="left")
        ytrue_col = "EF"
    else:
        ytrue_col = cols.get("ef")
        df = pred
        if "Split" not in df.columns:
            fl = pd.read_csv(os.path.join(data_dir, "FileList.csv"))
            df["__key__"] = df[fname_col].astype(str).str.replace(r"\.avi$","",regex=True)
            fl["__key__"] = fl["FileName"].astype(str).str.replace(r"\.avi$","",regex=True)
            df = df.merge(fl[["__key__","Split"]], on="__key__", how="left")

    # 只計算 TEST split
    if "Split" in df.columns:
        df = df[df["Split"].astype(str).str.upper()=="TEST"].copy()

    df = df.rename(columns={yhat_col:"PredEF", ytrue_col:"TrueEF"})
    df = df.dropna(subset=["PredEF","TrueEF"])
    return df

def compute_metrics(df):
    err = df["PredEF"] - df["TrueEF"]
    mae = err.abs().mean()
    rmse = np.sqrt((err**2).mean())
    # R^2（基於真值變異）
    r2 = 1 - ((err**2).sum() / ((df["TrueEF"]-df["TrueEF"].mean())**2).sum())
    return mae, rmse, r2

def plot_scatter(df, out_png):
    plt.figure()
    plt.scatter(df["TrueEF"], df["PredEF"], s=10)
    lims = [min(df["TrueEF"].min(), df["PredEF"].min())-2,
            max(df["TrueEF"].max(), df["PredEF"].max())+2]
    plt.plot(lims, lims)
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel("True EF (%)"); plt.ylabel("Predicted EF (%)")
    plt.title("EF Regression: Prediction vs Truth")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_bland_altman(df, out_png):
    diff = df["PredEF"] - df["TrueEF"]
    avg  = (df["PredEF"] + df["TrueEF"]) / 2
    m, s = diff.mean(), diff.std(ddof=1)
    plt.figure()
    plt.scatter(avg, diff, s=10)
    for y in [m, m+1.96*s, m-1.96*s]:
        plt.axhline(y, linestyle="--")
    plt.xlabel("Average EF (%)"); plt.ylabel("Pred - True (EF %)")
    plt.title("Bland–Altman")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_size_curves(size_csv, out_dir, n=3):
    if not size_csv or not os.path.exists(size_csv):
        return []
    df = pd.read_csv(size_csv)
    out_pngs = []
    for i, (k, g) in enumerate(df.groupby("FileName")):
        if i >= n: break
        plt.figure()
        if "Frame" in g.columns and "Area" in g.columns:
            plt.plot(g["Frame"], g["Area"])
            plt.xlabel("Frame"); plt.ylabel("LV area (px^2)")
            plt.title(str(k))
            png = os.path.join(out_dir, f"{re.sub(r'[^A-Za-z0-9_]+','_',str(k))}_area_curve.png")
            plt.tight_layout(); plt.savefig(png, dpi=150); plt.close()
            out_pngs.append(png)
    return out_pngs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--seg_dir", default="output/segmentation")
    ap.add_argument("--vid_dir", default="output/video")
    ap.add_argument("--n_curves", type=int, default=3)
    args = ap.parse_args()

    # 1) 找最新 video 輸出
    test_pred_csv = find_latest_csv(args.vid_dir, "test_predictions.csv")
    if test_pred_csv is None:
        raise SystemExit("❌ 找不到 test_predictions.csv，請先跑 `echonet video`")
    out_dir = os.path.dirname(test_pred_csv)
    ensure_dir(out_dir)

    # 2) 載入 & 計算指標
    df = load_test_df(test_pred_csv, args.data_dir)
    mae, rmse, r2 = compute_metrics(df)

    # 3) 繪圖
    scatter_png = os.path.join(out_dir, "scatter_pred_vs_true.png")
    bland_png   = os.path.join(out_dir, "bland_altman.png")
    plot_scatter(df, scatter_png)
    plot_bland_altman(df, bland_png)

    # 4) segmentation 曲線（若有）
    size_csv = find_latest_csv(args.seg_dir, "size.csv")
    curve_pngs = plot_size_curves(size_csv, out_dir, n=args.n_curves)

    # 5) 存摘要
    summ = os.path.join(out_dir, "metrics_summary.txt")
    with open(summ, "w", encoding="utf-8") as f:
        f.write(f"TEST N={len(df)}\nMAE={mae:.2f}\nRMSE={rmse:.2f}\nR2={r2:.3f}\n")
        f.write(f"pred_csv={test_pred_csv}\n")
        if size_csv:
            f.write(f"size_csv={size_csv}\n")
        f.write(f"scatter={scatter_png}\nbland_altman={bland_png}\n")
        if curve_pngs:
            f.write("curves=\n" + "\n".join(curve_pngs) + "\n")

    print("✅ Done")
    print(f"Saved: {summ}")
    print(f"Scatter: {scatter_png}")
    print(f"Bland–Altman: {bland_png}")
    if curve_pngs: print("Curves:", *curve_pngs, sep="\n - ")

if __name__ == "__main__":
    main()
