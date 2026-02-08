import argparse
import os
import re
from typing import Dict

import pandas as pd

from tqcenter import tq


def _validate_code(code: str) -> None:
    pattern = re.compile(r"^\d{6}\.[A-Z]{2,3}$")
    if not pattern.match(code):
        raise ValueError("stock code format must be 6 digits + suffix, e.g. 688318.SH")


def _validate_time(value: str) -> None:
    if not value:
        return
    if len(value) not in (8, 14) or not value.isdigit():
        raise ValueError("time format must be YYYYMMDD or YYYYMMDDHHMMSS")


def _dict_to_dataframe(data: Dict[str, pd.DataFrame], code: str) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()

    frames = []
    for field, df in data.items():
        if df is None or df.empty:
            continue
        if code not in df.columns:
            continue
        series = df[code].rename(field)
        frames.append(series)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, axis=1)
    result.index.name = "Date"
    result.insert(0, "Code", code)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Download forward-adjusted daily K line data.")
    parser.add_argument("--code", required=True, help="stock code, e.g. 688318.SH")
    parser.add_argument("--start", required=True, help="start time, YYYYMMDD or YYYYMMDDHHMMSS")
    parser.add_argument("--end", required=True, help="end time, YYYYMMDD or YYYYMMDDHHMMSS")
    parser.add_argument("--out-dir", default=".", help="output directory")
    args = parser.parse_args()

    _validate_code(args.code)
    _validate_time(args.start)
    _validate_time(args.end)

    tq.initialize(__file__)

    data = tq.get_market_data(
        field_list=[],
        stock_list=[args.code],
        period="1d",
        start_time=args.start,
        end_time=args.end,
        count=-1,
        dividend_type="front",
        fill_data=False,
    )

    df = _dict_to_dataframe(data, args.code)
    if df.empty:
        raise RuntimeError("no data returned, check client data and time range")

    # Skip non-trading days by dropping empty or invalid rows.
    if "Date" in df.columns:
        df = df[df["Date"].notna() & (df["Date"] != 0)]
    value_columns = [col for col in df.columns if col not in ("Code", "Date")]
    if value_columns:
        df = df.dropna(subset=value_columns, how="all")

    os.makedirs(args.out_dir, exist_ok=True)
    safe_start = args.start or ""
    safe_end = args.end or ""
    filename = f"{args.code}-{safe_start}-{safe_end}.csv"
    out_path = os.path.join(args.out_dir, filename)
    df.to_csv(out_path, index=True, encoding="utf-8-sig")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
