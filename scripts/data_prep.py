from datasets import load_dataset

ds = load_dataset("tasksource/esci", split="train")
# 1) locale == "us" filtrele
# 2) esci_label == "E" filtrele
# 3) her query_id için TEK bir positive seç  ← false negative önlemi
# 4) (query, product_title) çiftleri, ~50k
# 5) train/val ayır, parquet olarak kaydet