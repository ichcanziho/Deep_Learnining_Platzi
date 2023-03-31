import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding


def tokenize_fn(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


if __name__ == '__main__':
    # Descarga de dataset
    ds = load_dataset("glue", "mrpc")
    labels = ds["train"].features["label"].names
    # Descarga de tokenizador de DistilRoberta
    base_model_url = "distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(base_model_url)
    clean_ds = ds.map(tokenize_fn, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    import evaluate
    import numpy as np


    def compute_metrics(eval_pred):
        metric = evaluate.load("glue", "mrpc")
        logits, labels_ = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels_)


    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_url,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)}
    )

    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir="./platzi-distilroberta-base-mrpc-glue-gabriel-ichcanziho",
        evaluation_strategy="steps",
        num_train_epochs=4,
        push_to_hub_organization="platzi",
        push_to_hub=True,
        load_best_model_at_end=True
    )

    from transformers import Trainer

    trainer = Trainer(
        model,
        training_args,
        train_dataset=clean_ds["train"],
        eval_dataset=clean_ds["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
