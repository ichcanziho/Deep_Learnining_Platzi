from sklearn.model_selection import train_test_split
from conllu import parse_incr
from nltk.tag import hmm

data_file = open("../data/UD_Spanish-AnCora/es_ancora-ud-train.conllu", "r", encoding="utf-8")
data_array = []
for tokenlist in parse_incr(data_file):
    tokenized_text = []
    for token in tokenlist:
        tokenized_text.append((token['form'], token['upos']))
    data_array.append(tokenized_text)

print(data_array[:10])
print(len(data_array))

train_data, test_data = train_test_split(data_array, test_size=0.2, random_state=42)
print(len(train_data))
print(len(test_data))

tagger = hmm.HiddenMarkovModelTrainer().train_supervised(train_data)
print(tagger)

# Validación del modelo: Un vez entrenado el tagger, calcula el rendimiento del modelo (usando tagger.evaluate())
# para los conjuntos de entrenamiento y test.

train_acc = tagger.evaluate(train_data)
print("Accuracy over Train's partition:", train_acc)


test_acc = tagger.evaluate(test_data)
print("Accuracy over Test's partition:", test_acc)
