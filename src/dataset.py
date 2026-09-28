import pandas as pd
from torch.utils.data import DataLoader, Dataset


class PairDataset(Dataset):
    """(query, product_title) pairs from data_prep.py output."""

    def __init__(self, parquet_path):
        df = pd.read_parquet(parquet_path)
        self.queries = df["query"].tolist()
        self.docs = df["product_title"].tolist()

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx], self.docs[idx]


class PairCollator:
    """Tokenizes a whole batch at once; padding only to the longest item in the batch."""

    def __init__(self, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def _encode(self, texts):
        return self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_tensors="pt",
        )

    def __call__(self, batch):
        queries, docs = zip(*batch)
        return self._encode(queries), self._encode(docs)


def make_loader(parquet_path, tokenizer, batch_size=16, max_len=128, shuffle=True):
    return DataLoader(
        PairDataset(parquet_path),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,  # a smaller last batch = fewer in-batch negatives
        collate_fn=PairCollator(tokenizer, max_len),
        num_workers=0,
        pin_memory=True,
    )
