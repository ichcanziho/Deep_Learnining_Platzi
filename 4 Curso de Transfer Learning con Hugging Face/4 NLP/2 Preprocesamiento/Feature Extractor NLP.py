import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datasets import load_dataset

ds = load_dataset("glue", "mrpc")
labels = ds["train"].features["label"].names

from transformers import AutoTokenizer

base_model_url = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(base_model_url)

ejemplo = ds["train"][400]
print(ejemplo)
print(tokenizer(ds["train"][400]["sentence1"]))


def tokenize_fn(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


clean_ds = ds.map(tokenize_fn, batched=True)
print(clean_ds["train"][400])

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print(data_collator)
