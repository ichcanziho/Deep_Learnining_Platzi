import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import evaluate
import numpy as np


def tokenize_fn(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


def compute_metrics(eval_pred):
    metric = evaluate.load("glue", "mrpc")
    logits, labels_ = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels_)


if __name__ == '__main__':

    # 1. Descarga de dataset
    ds = load_dataset("glue", "mrpc")
    labels = ds["train"].features["label"].names

    # 2. Descarga de tokenizador de DistilRoberta
    base_model_url = "distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(base_model_url)

    # 3. Limpiamos la base de datos original
    clean_ds = ds.map(tokenize_fn, batched=True)

    # 4. Creamos nuestro DataCollator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 5. Definimos el modelo
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_url,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)}
    )

    # 6. Definimos los argumentos de entrenamiento del modelo
    training_args = TrainingArguments(
        output_dir="./platzi-distilroberta-base-mrpc-glue-gabriel-ichcanziho",
        evaluation_strategy="steps",
        num_train_epochs=4,
        push_to_hub_organization="platzi",
        push_to_hub=True,
        load_best_model_at_end=True
    )

    # 7. Definimos al propio entrenador
    trainer = Trainer(
        model,
        training_args,
        train_dataset=clean_ds["train"],
        eval_dataset=clean_ds["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # 8. Entrenamos al modelo
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    print("---------------TEST----------------")
    # 9. Evaluando en el conjunto de test
    metrics = trainer.evaluate(clean_ds["test"])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
