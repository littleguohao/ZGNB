import glob
import os
from typing import Dict, List

import numpy as np
import pandas as pd


WINDOW = 50
FACTOR_COLS = [
    "VAM50",
    "REV50",
    "LVOL50",
    "PVDC50",
    "MSI50",
    "BreakoutFollow_50",
    "DrawdownPressure_50",
    "VolatilityExpansion_50",
    "PriceVolumeCoMove_50",
    "UpperWickRatio_50",
    "OpenCloseDominance_50",
    "VolumeImpulse_50",
]


def rolling_rank_last(series: pd.Series, window: int) -> pd.Series:
    def _rank_last(values: np.ndarray) -> float:
        s = pd.Series(values)
        return s.rank(pct=True).iloc[-1]

    return series.rolling(window).apply(_rank_last, raw=True)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return (series - mean) / std


def load_csvs(data_dir: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in glob.glob(os.path.join(data_dir, "*.csv")):
        base = os.path.basename(path)
        if base.startswith("factor_"):
            continue
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        if "Date" not in df.columns or "Code" not in df.columns:
            continue
        required = {"Open", "Close", "High", "Low", "Volume"}
        if not required.issubset(set(df.columns)):
            continue
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    out_frames: List[pd.DataFrame] = []
    for code, g in df.groupby("Code", sort=False):
        g = g.sort_values("Date").copy()

        open_ = g["Open"].astype(float)
        close = g["Close"].astype(float)
        high = g["High"].astype(float)
        low = g["Low"].astype(float)
        vol = g["Volume"].astype(float)

        ret = close.pct_change(fill_method=None)
        ret10 = ret.rolling(10).sum()
        ret20 = ret.rolling(20).sum()
        range_ratio = (high - low) / close
        dvol = vol.pct_change(fill_method=None)
        dclose = close.diff()
        dvol_abs = vol.diff()

        prev_high_50 = high.shift(1).rolling(WINDOW, min_periods=WINDOW).max()
        max_close_50 = close.rolling(WINDOW, min_periods=WINDOW).max()
        std50 = ret.rolling(WINDOW, min_periods=WINDOW).std(ddof=0)
        vol_ma50 = vol.rolling(WINDOW, min_periods=WINDOW).mean()
        range_den = (high - low).replace(0, np.nan)

        vam50 = rolling_zscore(vol, WINDOW) * rolling_zscore(ret10, WINDOW)
        rev50 = -rolling_zscore(ret20, WINDOW)
        lvol50 = -ret.rolling(WINDOW, min_periods=WINDOW).std(ddof=0)
        pvdc50 = -ret.rolling(WINDOW, min_periods=WINDOW).corr(dvol)
        msi50 = rolling_zscore(range_ratio * vol, WINDOW)

        breakout_follow_50 = ((close - prev_high_50) / prev_high_50) * (vol / vol_ma50)
        drawdown_pressure_50 = (max_close_50 - close) / max_close_50
        volatility_expansion_50 = std50.diff()
        price_volume_comove_50 = dclose.rolling(WINDOW, min_periods=WINDOW).corr(dvol_abs)
        upper_wick_ratio_50 = ((high - np.maximum(close, open_)) / range_den).rolling(
            WINDOW, min_periods=WINDOW
        ).mean()
        open_close_dominance_50 = ((close - open_) / range_den).rolling(
            WINDOW, min_periods=WINDOW
        ).mean()
        volume_impulse_50 = vol / vol_ma50

        out = pd.DataFrame(
            {
                "Date": g["Date"],
                "Code": code,
                "VAM50": vam50,
                "REV50": rev50,
                "LVOL50": lvol50,
                "PVDC50": pvdc50,
                "MSI50": msi50,
                "BreakoutFollow_50": breakout_follow_50,
                "DrawdownPressure_50": drawdown_pressure_50,
                "VolatilityExpansion_50": volatility_expansion_50,
                "PriceVolumeCoMove_50": price_volume_comove_50,
                "UpperWickRatio_50": upper_wick_ratio_50,
                "OpenCloseDominance_50": open_close_dominance_50,
                "VolumeImpulse_50": volume_impulse_50,
            }
        )
        out_frames.append(out)

    return pd.concat(out_frames, ignore_index=True)


def summarize_by_stock(panel: pd.DataFrame, factor_cols: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for code, g in panel.groupby("Code", sort=False):
        row: Dict[str, float] = {"Code": code}
        for col in factor_cols:
            series = g[col].dropna()
            row[f"{col}_Mean"] = series.mean() if not series.empty else np.nan
            row[f"{col}_Std"] = series.std(ddof=0) if not series.empty else np.nan
            row[f"{col}_Last"] = series.iloc[-1] if not series.empty else np.nan
            row[f"{col}_Count"] = float(series.count())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "B1_DATA")
    out_dir = os.path.join(base_dir, "R_DATA")
    panel_out = os.path.join(out_dir, "factor_panel.csv")
    summary_out = os.path.join(out_dir, "factor_stock_summary.csv")

    raw = load_csvs(data_dir)
    if raw.empty:
        raise RuntimeError("no csv files found in B1_DATA")

    os.makedirs(out_dir, exist_ok=True)

    panel = compute_factors(raw)
    panel.to_csv(panel_out, index=False, encoding="utf-8-sig")

    summary = summarize_by_stock(panel, FACTOR_COLS)
    summary.to_csv(summary_out, index=False, encoding="utf-8-sig")

    print("saved:", panel_out)
    print("saved:", summary_out)


if __name__ == "__main__":
    main()
