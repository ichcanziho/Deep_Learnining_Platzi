import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datasets import load_dataset
from transformers import ViTFeatureExtractor
import torch


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch])
    }


def transform(example_batch):
    inputs = feature_extractor([x for x in example_batch["image"]], return_tensors="pt")
    inputs["labels"] = example_batch["labels"]
    return inputs


if __name__ == '__main__':
    ds = load_dataset("beans")

    labels = ds["train"].features["labels"].names
    print(labels)

    base_model_url = 'google/vit-base-patch16-224-in21k'
    feature_extractor = ViTFeatureExtractor.from_pretrained(base_model_url)
    clean = ds.with_transform(transform)

    import numpy as np
    from datasets import load_metric

    metric = load_metric("accuracy")


    def compute_metrics(prediction):
        return metric.compute(predictions=np.argmax(prediction.predictions, axis=1), references=prediction.label_ids)

    from transformers import ViTForImageClassification

    model = ViTForImageClassification.from_pretrained(
        base_model_url,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)}
    )

    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir="./platzi-vit-model-gabriel-ichcanziho",
        evaluation_strategy="steps",
        num_train_epochs=4,
        push_to_hub_organization="platzi",
        learning_rate=2e-4,
        remove_unused_columns=False,
        push_to_hub=True,
        load_best_model_at_end=True,
    )

    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=clean["train"],
        eval_dataset=clean["validation"],
        tokenizer=feature_extractor
    )