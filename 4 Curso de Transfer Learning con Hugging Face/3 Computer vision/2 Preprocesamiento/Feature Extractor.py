import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datasets import load_dataset

ds = load_dataset("beans")
ex = ds["train"][400]
test_image = ex["image"]

labels = ds["train"].features["labels"]
label_name = labels.int2str(ex["labels"])

from transformers import ViTFeatureExtractor

base_model_url = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(base_model_url)
print(feature_extractor)

feature_test = feature_extractor(test_image, return_tensors="pt")
print(feature_test)
print(feature_test.keys())
print(feature_test["pixel_values"].shape)


def process_example(example):
    inputs = feature_extractor(example["image"], return_tensors="pt")
    inputs["labels"] = example["labels"]
    return inputs


preprocess_test = ds["train"][10]
print(preprocess_test)
preprocess_test = process_example(preprocess_test)
print("cleaned")
print(preprocess_test)
print(preprocess_test["pixel_values"].shape)


def transform(example_batch):
    inputs = feature_extractor([x for x in example_batch["image"]], return_tensors="pt")
    inputs["labels"] = example_batch["labels"]
    return inputs


print("batch")
clean = ds.with_transform(transform)
print(clean["train"][0:2])
print(clean["train"][0:2]["pixel_values"].shape)


import torch


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch])
    }
