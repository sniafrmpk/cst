#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run delta fits for D=2,3 (and any others you add) at H=1.
Infers N for N0k files using target label at t==H.
"""
import os
import re
import sys
import importlib.util
import numpy as np
import pandas as pd


def load_module(path):
    spec = importlib.util.spec_from_file_location("delta_dec", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["delta_dec"] = mod
    spec.loader.exec_module(mod)
    return mod


def infer_n_from_target(file_path, H):
    df = pd.read_csv(file_path, usecols=[0, 1, 2])
    counts = df.loc[df["t"] == H, "element"].value_counts()
    if len(counts) == 0:
        return None
    target_label = int(counts.index[0])
    return target_label - 1


def file_entries_from_folder(folder, H):
    entries = []
    for fname in os.listdir(folder):
        if not fname.endswith(".csv"):
            continue
        file_path = os.path.join(folder, fname)
        if os.path.getsize(file_path) == 0:
            continue
        m = re.search(r"_N(\d+)k", fname)
        if not m:
            continue
        k = int(m.group(1))
        if k > 0:
            N = k * 1000
        else:
            # Prefer explicit N in filename (e.g., "N0k 654.csv")
            m2 = re.search(r"_N0k\s*(\d+)", fname)
            if m2:
                N = int(m2.group(1))
            else:
                N = infer_n_from_target(file_path, H)
                if N is None:
                    continue
        entries.append((N, file_path))

    # Deduplicate by N; keep the larger file if duplicates exist
    dedup = {}
    for N, path in entries:
        size = os.path.getsize(path)
        if N not in dedup or size > dedup[N][1]:
            dedup[N] = (path, size)
    return sorted((N, path) for N, (path, _) in dedup.items())


def fit(df, D):
    df = df[df["D"] == D]
    results = {}
    for H, group in df.groupby("height"):
        N = group["N"].values
        y = (group["delta"] / group["height"]).values
        X = np.log(N)
        Y = np.log(y)
        slope, intercept = np.polyfit(X, Y, 1)
        beta = -slope
        A = np.exp(intercept)
        results[H] = {"A": A, "beta": beta}
    return results


def main():
    script_path = "Data analysis/Delta analysis - Dec.py"
    mod = load_module(script_path)

    for D in [2, 3]:
        H = 1
        folder = mod.read_path(D, H)
        entries = file_entries_from_folder(folder, H)
        n_values = [n for n, _ in entries]
        print(f"\nD={D}, H={H} | N values: {n_values}")

        delta_data_mid = []
        delta_data_all = []
        for N, file_path in entries:
            _, delta_data_mid_n = mod.delta_analyzer_middle_layers(
                folder, D, H, N, True, file_path_override=file_path
            )
            _, delta_data_all_n = mod.delta_analyzer_middle_layers(
                folder, D, H, N, False, file_path_override=file_path
            )
            delta_data_mid.append(delta_data_mid_n)
            delta_data_all.append(delta_data_all_n)

        df_mid = pd.DataFrame(delta_data_mid, columns=[
            "D", "height", "N", "rho", "l0", "delta",
            "delta_over_H", "delta_over_l0", "height_over_l0"
        ])
        df_all = pd.DataFrame(delta_data_all, columns=[
            "D", "height", "N", "rho", "l0", "delta",
            "delta_over_H", "delta_over_l0", "height_over_l0"
        ])

        mid_fit = fit(df_mid, D)
        all_fit = fit(df_all, D)

        print("Middle-third fit:", mid_fit)
        print("All-layers fit:", all_fit)

        # Generate plots
        mod.δbyH_vs_N([D], df_mid, delta_definition=True)
        mod.δbyH_vs_N([D], df_all, delta_definition=False)

        for H in sorted(set(mid_fit.keys()) | set(all_fit.keys())):
            m = mid_fit.get(H)
            a = all_fit.get(H)
            print(
                f"H={H} | middle: A={m['A']:.6g}, beta={m['beta']:.6g} | "
                f"all: A={a['A']:.6g}, beta={a['beta']:.6g} | "
                f"ratio A_mid/all={m['A']/a['A']:.6g}, delta_beta={m['beta']-a['beta']:.6g}"
            )


if __name__ == "__main__":
    main()
