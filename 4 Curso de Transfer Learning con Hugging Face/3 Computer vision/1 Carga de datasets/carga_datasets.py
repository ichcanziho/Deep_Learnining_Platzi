from datasets import load_dataset
import matplotlib.pyplot as plt

ds = load_dataset("beans")

print(ds)

# Mostremos un ejemplo:
ex = ds["train"][400]
print(ex)

plt.imshow(ex["image"])
plt.savefig("ejemplo.png")
plt.close()

labels = ds["train"].features["labels"]
print(labels)

label_name = labels.int2str(ex["labels"])
print(label_name)
