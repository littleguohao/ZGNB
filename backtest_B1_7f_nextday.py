import argparse
import math
import os
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from tqcenter import tq


WINDOW = 50
KDJ_N = 9
KDJ_ALPHA = 1 / 3
FACTOR_SET_7 = [
    "BreakoutFollow_50",
    "DrawdownPressure_50",
    "VolatilityExpansion_50",
    "PriceVolumeCoMove_50",
    "UpperWickRatio_50",
    "OpenCloseDominance_50",
    "VolumeImpulse_50",
]


def chunk_list(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def infer_reference_codes(data_dir: str) -> List[str]:
    codes = []
    if not os.path.isdir(data_dir):
        return codes
    for name in os.listdir(data_dir):
        if not name.endswith(".csv"):
            continue
        if name.startswith("factor_") or name.startswith("B1_"):
            continue
        code = name.split("-")[0]
        if code and code not in codes:
            codes.append(code)
    return sorted(codes)


def fetch_ohlcv_history(
    stock_list: List[str],
    start_time: str,
    end_time: str,
) -> Dict[str, pd.DataFrame]:
    fields = ["Open", "High", "Low", "Close", "Volume"]
    field_frames: Dict[str, List[pd.DataFrame]] = {f: [] for f in fields}

    trade_dates = tq.get_trading_dates(market="SH", start_time=start_time, end_time=end_time, count=0)
    count = len(trade_dates) if isinstance(trade_dates, list) else 240
    count = max(60, count)
    chunk_size = max(1, 24000 // max(1, count))

    for chunk in chunk_list(stock_list, chunk_size):
        data = tq.get_market_data(
            field_list=fields,
            stock_list=chunk,
            start_time=start_time,
            end_time=end_time,
            count=count,
            dividend_type="front",
            period="1d",
            fill_data=True,
        )
        for field in fields:
            df_field = tq.price_df(data, field, column_names=chunk)
            field_frames[field].append(df_field)

    result: Dict[str, pd.DataFrame] = {}
    for field in fields:
        if not field_frames[field]:
            result[field] = pd.DataFrame()
        else:
            df = pd.concat(field_frames[field], axis=1)
            df = df.sort_index()
            result[field] = df
    return result


def compute_kdj_j(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    low_n = low.rolling(KDJ_N, min_periods=KDJ_N).min()
    high_n = high.rolling(KDJ_N, min_periods=KDJ_N).max()
    rsv = (close - low_n) / (high_n - low_n) * 100
    k = rsv.ewm(alpha=KDJ_ALPHA, adjust=False).mean()
    d = k.ewm(alpha=KDJ_ALPHA, adjust=False).mean()
    j = 3 * k - 2 * d
    return j


def compute_factors(open_: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, vol: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    ret = close.pct_change(fill_method=None)
    std50 = ret.rolling(WINDOW, min_periods=WINDOW).std(ddof=0)
    vol_ma50 = vol.rolling(WINDOW, min_periods=WINDOW).mean()
    range_den = (high - low).replace(0, np.nan)

    prev_high_50 = high.shift(1).rolling(WINDOW, min_periods=WINDOW).max()
    max_close_50 = close.rolling(WINDOW, min_periods=WINDOW).max()

    breakout_follow_50 = ((close - prev_high_50) / prev_high_50) * (vol / vol_ma50)
    drawdown_pressure_50 = (max_close_50 - close) / max_close_50
    volatility_expansion_50 = std50.diff()

    # PVC: compute per column to avoid heavy multi-index output
    pvc_cols = {}
    for col in close.columns:
        pvc_cols[col] = close[col].diff().rolling(WINDOW, min_periods=WINDOW).corr(vol[col].diff())
    price_volume_comove_50 = pd.DataFrame(pvc_cols, index=close.index)

    upper_wick_ratio_50 = ((high - np.maximum(close, open_)) / range_den).rolling(
        WINDOW, min_periods=WINDOW
    ).mean()
    open_close_dominance_50 = ((close - open_) / range_den).rolling(
        WINDOW, min_periods=WINDOW
    ).mean()
    volume_impulse_50 = vol / vol_ma50

    return {
        "BreakoutFollow_50": breakout_follow_50,
        "DrawdownPressure_50": drawdown_pressure_50,
        "VolatilityExpansion_50": volatility_expansion_50,
        "PriceVolumeCoMove_50": price_volume_comove_50,
        "UpperWickRatio_50": upper_wick_ratio_50,
        "OpenCloseDominance_50": open_close_dominance_50,
        "VolumeImpulse_50": volume_impulse_50,
    }


def zscore_series(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - series.mean()) / std


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest B1 factor-set 7 with next-day returns.")
    parser.add_argument("--start", required=True, help="YYYYMMDD")
    parser.add_argument("--end", default="", help="YYYYMMDD (blank = latest)")
    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "R_DATA"))
    parser.add_argument("--ref-dir", default=os.path.join(os.path.dirname(__file__), "B1_DATA"))
    args = parser.parse_args()

    tq.initialize(__file__)

    pool_codes = tq.get_stock_list("5")
    if not pool_codes:
        raise RuntimeError("empty pool from TDX")

    ref_codes = infer_reference_codes(args.ref_dir)
    if not ref_codes:
        raise RuntimeError("reference codes not found")

    ohlcv = fetch_ohlcv_history(pool_codes, args.start, args.end)
    open_df = ohlcv["Open"]
    high_df = ohlcv["High"]
    low_df = ohlcv["Low"]
    close_df = ohlcv["Close"]
    vol_df = ohlcv["Volume"]

    start_dt = pd.to_datetime(args.start, format="%Y%m%d")
    end_dt = pd.to_datetime(args.end, format="%Y%m%d") if args.end else None
    open_df.index = pd.to_datetime(open_df.index)
    high_df.index = pd.to_datetime(high_df.index)
    low_df.index = pd.to_datetime(low_df.index)
    close_df.index = pd.to_datetime(close_df.index)
    vol_df.index = pd.to_datetime(vol_df.index)
    if end_dt is not None:
        mask = (close_df.index >= start_dt) & (close_df.index <= end_dt)
    else:
        mask = close_df.index >= start_dt
    open_df = open_df.loc[mask]
    high_df = high_df.loc[mask]
    low_df = low_df.loc[mask]
    close_df = close_df.loc[mask]
    vol_df = vol_df.loc[mask]

    j_df = compute_kdj_j(high_df, low_df, close_df)
    factor_map = compute_factors(open_df, high_df, low_df, close_df, vol_df)

    dates = close_df.index
    results = []
    picks = []

    ref_codes = [c for c in ref_codes if c in close_df.columns]
    if not ref_codes:
        raise RuntimeError("reference codes not in pool")

    for idx in range(len(dates) - 1):
        date = dates[idx]
        next_date = dates[idx + 1]
        if date < start_dt:
            continue
        if end_dt is not None and date > end_dt:
            continue

        j_today = j_df.loc[date]
        candidates = j_today[j_today < 13].index.tolist()
        if not candidates:
            continue

        union = sorted(set(candidates).union(ref_codes))
        factor_rows = {}
        for name in FACTOR_SET_7:
            series = factor_map[name].loc[date, union]
            factor_rows[name] = series
        factor_df = pd.DataFrame(factor_rows)

        z_df = factor_df.apply(zscore_series, axis=0)
        ref_mean = z_df.loc[ref_codes].mean(axis=0)

        sim_rows = []
        for code in candidates:
            vec = z_df.loc[code].to_numpy(dtype=float)
            mask = ~np.isnan(vec) & ~np.isnan(ref_mean.to_numpy(dtype=float))
            if mask.sum() < 3:
                continue
            sim = cosine_similarity(vec[mask], ref_mean.to_numpy(dtype=float)[mask])
            sim_rows.append((code, sim))

        if not sim_rows:
            continue

        sim_df = pd.DataFrame(sim_rows, columns=["Code", "Similarity"]).sort_values(
            "Similarity", ascending=False
        )
        top = sim_df.head(args.topn)
        for code in top["Code"].tolist():
            picks.append(
                {
                    "Date": date.strftime("%Y-%m-%d"),
                    "NextDate": next_date.strftime("%Y-%m-%d"),
                    "Code": code,
                }
            )
        next_ret = close_df.loc[next_date, top["Code"]] / close_df.loc[date, top["Code"]] - 1
        avg_ret = float(next_ret.mean())
        win_rate = float((next_ret > 0).mean())

        results.append(
            {
                "Date": date.strftime("%Y-%m-%d"),
                "NextDate": next_date.strftime("%Y-%m-%d"),
                "Count": len(top),
                "AvgNextRet": avg_ret,
                "WinRate": win_rate,
            }
        )

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "B1_backtest_7f_nextday.csv")
    picks_path = os.path.join(out_dir, "B1_backtest_7f_picks.csv")
    summary = pd.DataFrame(results)
    if not summary.empty:
        summary["Date"] = pd.to_datetime(summary["Date"], errors="coerce")
        summary = summary[summary["Date"] >= start_dt]
        if end_dt is not None:
            summary = summary[summary["Date"] <= end_dt]
        summary["Date"] = summary["Date"].dt.strftime("%Y-%m-%d")
    summary.to_csv(out_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(picks).to_csv(picks_path, index=False, encoding="utf-8-sig")
    summary_path = os.path.join(out_dir, "B1_backtest_7f_summary.csv")
    if not summary.empty:
        stats = {
            "Days": len(summary),
            "AvgNextRet": float(summary["AvgNextRet"].mean()),
            "WinRate": float((summary["AvgNextRet"] > 0).mean()),
        }
        pd.DataFrame([stats]).to_csv(summary_path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame([{"Days": 0, "AvgNextRet": np.nan, "WinRate": np.nan}]).to_csv(
            summary_path, index=False, encoding="utf-8-sig"
        )
    print("saved:", out_path)
    print("saved:", summary_path)
    print("saved:", picks_path)


if __name__ == "__main__":
    main()
