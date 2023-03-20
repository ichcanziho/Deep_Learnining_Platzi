import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


train = pd.read_csv("../datasets/sign_mnist_train/sign_mnist_train_clean.csv")
print(train)
y = train[["label"]]
n_clases = sorted(y["label"].unique())
print(len(n_clases), n_clases)
# Data visualization

fig, ax = plt.subplots(figsize=(10, 10))
sns.countplot(y, x="label")
ax.bar_label(ax.containers[0], rotation=45, label_type="edge", fmt=lambda x: '{:.1f}%'.format(x/len(train) * 100))
plt.xlabel("Classes", size=15)
plt.ylabel("Frequency", size=15)
plt.title("Data distribution")
plt.savefig("freq.png")

# Data Cleaning

X = train.drop(columns=["label"])

print(X.info)

print(X.dtypes)

print(X.isnull().values.any())

print(X[X.duplicated()])

print(X[X['pixel1'] == "fwefew"])

X = X.drop([595, 689, 727, 802, 861], axis=0)
X = X.astype(str).astype(int)
X /= 255
print(X.head())
print(X.dtypes)
