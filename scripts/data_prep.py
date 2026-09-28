from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path(__file__).resolve().parent.parent / "data"
N_PAIRS = 50_000
VAL_FRAC = 0.05
SEED = 42
COLUMNS = ["query_id", "query", "product_locale", "esci_label", "product_title"]


def build_pairs(split):
    ds = load_dataset("tasksource/esci", split=split).select_columns(COLUMNS)
    df = ds.to_pandas()
    print(f"[{split}] raw rows: {len(df):,}")

    # 1) US + 2) Exact
    df = df[(df.product_locale == "us") & (df.esci_label == "Exact")]
    df = df.dropna(subset=["query", "product_title"])
    df = df[df.product_title.str.strip() != ""]
    print(f"[{split}] us & E: {len(df):,} rows, {df.query_id.nunique():,} queries")

    # 3) One positive per query_id
    df = df.groupby("query_id").sample(n=1, random_state=SEED)

    # Duplicate texts in a batch become false negatives
    df = df.drop_duplicates("query").drop_duplicates("product_title")
    print(f"[{split}] after dedup: {len(df):,} pairs")

    return df[["query_id", "query", "product_title"]].reset_index(drop=True)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    # 4) ~50k (query, product_title) pairs
    pairs = build_pairs("train")
    pairs = pairs.sample(n=min(N_PAIRS, len(pairs)), random_state=SEED).reset_index(drop=True)

    # 5) train/val split (one row per query -> no query overlap)
    n_val = int(len(pairs) * VAL_FRAC)
    val, train = pairs.iloc[:n_val], pairs.iloc[n_val:]

    train.to_parquet(OUT_DIR / "train.parquet", index=False)
    val.to_parquet(OUT_DIR / "val.parquet", index=False)
    print(f"saved train={len(train):,}  val={len(val):,}  -> {OUT_DIR}")
    print(train.head(3).to_string())


if __name__ == "__main__":
    main()