import argparse
import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


WINDOW = 50
TOPN = 10
MIN_COUNT = 20
FACTOR_COLS = ["VAM50", "REV50", "LVOL50", "PVDC50", "MSI50"]


def load_panel(data_dir: str) -> pd.DataFrame:
    panel_path = os.path.join(data_dir, "factor_panel.csv")
    if not os.path.exists(panel_path):
        raise FileNotFoundError("factor_panel.csv not found. Run factor_batch.py first.")
    df = pd.read_csv(panel_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values(["Code", "Date"])
    return df


def infer_codes_from_csv(data_dir: str) -> List[str]:
    codes = []
    for path in glob.glob(os.path.join(data_dir, "*.csv")):
        base = os.path.basename(path)
        if base.startswith("factor_"):
            continue
        code = base.split("-")[0]
        if code and code not in codes:
            codes.append(code)
    return sorted(codes)


def load_reference_codes(data_dir: str, ref_codes: str, ref_file: str) -> List[str]:
    if ref_codes:
        return [c.strip() for c in ref_codes.split(",") if c.strip()]
    if ref_file:
        with open(ref_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    ref_path = os.path.join(data_dir, "reference_codes.txt")
    if os.path.exists(ref_path):
        with open(ref_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    codes = infer_codes_from_csv(data_dir)
    return codes[:10]


def last_window_series(g: pd.DataFrame, col: str, window: int) -> pd.Series:
    s = g[col].dropna()
    if len(s) < MIN_COUNT:
        return pd.Series(dtype=float)
    return s.iloc[-window:]


def zscore_series(s: pd.Series) -> pd.Series:
    if s.empty:
        return s
    mean = s.mean()
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(dtype=float)
    return (s - mean) / std


def build_reference_template(panel: pd.DataFrame, ref_codes: List[str]) -> Dict[str, pd.Series]:
    template: Dict[str, pd.Series] = {}
    for col in FACTOR_COLS:
        series_list = []
        for code in ref_codes:
            g = panel[panel["Code"] == code]
            s = last_window_series(g, col, WINDOW)
            if s.empty:
                continue
            z = zscore_series(s)
            if z.empty:
                continue
            series_list.append(z.reset_index(drop=True))
        if len(series_list) < 2:
            template[col] = pd.Series(dtype=float)
            continue
        aligned = pd.concat(series_list, axis=1)
        template[col] = aligned.mean(axis=1)
    return template


def vectorize_stock(panel: pd.DataFrame, code: str) -> np.ndarray:
    g = panel[panel["Code"] == code]
    vec = []
    for col in FACTOR_COLS:
        s = last_window_series(g, col, WINDOW)
        if s.empty:
            vec.extend([np.nan, np.nan, np.nan])
            continue
        vec.extend([s.mean(), s.std(ddof=0), s.iloc[-1]])
    return np.array(vec, dtype=float)


def similarity_euclid(panel: pd.DataFrame, ref_codes: List[str], candidates: List[str]) -> pd.DataFrame:
    ref_vecs = []
    for code in ref_codes:
        v = vectorize_stock(panel, code)
        if np.isnan(v).all():
            continue
        ref_vecs.append(v)
    if not ref_vecs:
        return pd.DataFrame(columns=["Code", "Distance"])
    ref_mat = np.stack(ref_vecs, axis=0)
    col_mask = ~np.all(np.isnan(ref_mat), axis=0)
    if not col_mask.any():
        return pd.DataFrame(columns=["Code", "Distance"])
    ref_vec = np.nanmean(ref_mat[:, col_mask], axis=0)
    if np.isnan(ref_vec).all():
        return pd.DataFrame(columns=["Code", "Distance"])

    rows = []
    for code in candidates:
        v = vectorize_stock(panel, code)[col_mask]
        mask = ~np.isnan(ref_vec) & ~np.isnan(v)
        if mask.sum() < 5:
            continue
        dist = np.linalg.norm(v[mask] - ref_vec[mask])
        rows.append({"Code": code, "Distance": dist})
    if not rows:
        return pd.DataFrame(columns=["Code", "Distance"])
    df = pd.DataFrame(rows).sort_values("Distance")
    return df


def similarity_corr(panel: pd.DataFrame, ref_codes: List[str], candidates: List[str]) -> pd.DataFrame:
    template = build_reference_template(panel, ref_codes)
    rows = []
    for code in candidates:
        g = panel[panel["Code"] == code]
        sims = []
        for col in FACTOR_COLS:
            tmpl = template.get(col, pd.Series(dtype=float))
            if tmpl.empty:
                continue
            s = last_window_series(g, col, WINDOW)
            if s.empty:
                continue
            z = zscore_series(s)
            if z.empty:
                continue
            z = z.reset_index(drop=True)
            m = min(len(z), len(tmpl))
            if m < MIN_COUNT:
                continue
            corr = pd.Series(z.iloc[-m:]).corr(pd.Series(tmpl.iloc[-m:]))
            if not np.isnan(corr):
                sims.append(corr)
        if sims:
            rows.append({"Code": code, "Similarity": float(np.mean(sims))})
    if not rows:
        return pd.DataFrame(columns=["Code", "Similarity"])
    df = pd.DataFrame(rows).sort_values("Similarity", ascending=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Select pool stocks similar to reference set.")
    base_dir = os.path.dirname(__file__)
    parser.add_argument("--data-dir", default=os.path.join(base_dir, "R_DATA"))
    parser.add_argument("--ref-codes", default="", help="comma-separated reference codes")
    parser.add_argument("--ref-file", default="", help="path to reference code list")
    parser.add_argument("--ref-dir", default=os.path.join(base_dir, "B1_DATA"))
    parser.add_argument("--topn", type=int, default=TOPN)
    parser.add_argument("--include-ref", action="store_true")
    args = parser.parse_args()

    panel = load_panel(args.data_dir)
    ref_codes = load_reference_codes(args.ref_dir, args.ref_codes, args.ref_file)
    if not ref_codes:
        raise RuntimeError("no reference codes found")

    all_codes = sorted(panel["Code"].dropna().unique().tolist())
    ref_codes = [c for c in ref_codes if c in set(all_codes)]
    if not ref_codes:
        raise RuntimeError("reference codes not found in factor_panel.csv")
    if args.include_ref:
        candidates = all_codes
    else:
        candidates = [c for c in all_codes if c not in set(ref_codes)]
    if not candidates:
        print("no candidates available. add pool CSVs or use --include-ref")

    sim1 = similarity_euclid(panel, ref_codes, candidates)
    sim2 = similarity_corr(panel, ref_codes, candidates)

    out1 = os.path.join(args.data_dir, "similarity_euclid.csv")
    out2 = os.path.join(args.data_dir, "similarity_corr.csv")
    sim1.to_csv(out1, index=False, encoding="utf-8-sig")
    sim2.to_csv(out2, index=False, encoding="utf-8-sig")

    print("reference codes:", ",".join(ref_codes))
    print("saved:", out1)
    print("saved:", out2)
    if not sim1.empty:
        print("top euclid:")
        print(sim1.head(args.topn).to_string(index=False))
    else:
        print("top euclid: empty")
    if not sim2.empty:
        print("top corr:")
        print(sim2.head(args.topn).to_string(index=False))
    else:
        print("top corr: empty")


if __name__ == "__main__":
    main()
