from datasets import load_dataset

ds = load_dataset("tasksource/esci", split="train")
print(ds)
print(ds[0])