import os
import pandas as pd


def load_top10(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["Code", "Rank"])
    df = pd.read_csv(path)
    if "Code" not in df.columns:
        return pd.DataFrame(columns=["Code", "Rank"])
    df = df[["Code"]].copy()
    df["Rank"] = range(1, len(df) + 1)
    return df


def main() -> None:
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "R_DATA")

    paths = {
        "5f": os.path.join(data_dir, "B1_top10_5f.csv"),
        "7f": os.path.join(data_dir, "B1_top10_7f.csv"),
        "12f": os.path.join(data_dir, "B1_top10_12f.csv"),
    }

    frames = []
    for key, path in paths.items():
        df = load_top10(path)
        df = df.rename(columns={"Rank": f"Rank_{key}"})
        frames.append(df)

    if not frames:
        raise RuntimeError("no top10 files found")

    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="Code", how="outer")

    merged = merged.sort_values(["Rank_5f", "Rank_7f", "Rank_12f"], na_position="last")
    out_path = os.path.join(data_dir, "B1_top10_compare.csv")
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("saved:", out_path)


if __name__ == "__main__":
    main()
