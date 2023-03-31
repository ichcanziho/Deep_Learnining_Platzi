from datasets import load_dataset

ds = load_dataset("glue", "mrpc")
print(ds)

# Mostremos un ejemplo:
ejemplo = ds["train"][400]
print(ejemplo)
labels = ds["train"].features["label"]
print(labels)
# Nombre de la etiqueta
label_name = labels.int2str(ejemplo["label"])
print(label_name)
