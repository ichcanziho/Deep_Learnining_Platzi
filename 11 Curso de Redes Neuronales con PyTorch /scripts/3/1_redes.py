import torch
import torchtext
from torchtext.datasets import DBpedia
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader

# Comprobar la versión
print(torchtext.__version__)

train_iter = iter(DBpedia(split="train"))
print(next(train_iter))
print(next(train_iter))

tokenizador = get_tokenizer("basic_english")
train_iter = DBpedia(split="train")


def yield_tokens(data_iter):
    for _, texto in data_iter:
        yield tokenizador(texto)


vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

ans = vocab(tokenizador("Hello how are you? I am a platzi student"))
print(ans)

texto_pipeline = lambda x: vocab(tokenizador(x))
label_pipeline = lambda x: int(x) - 1
ans = texto_pipeline("Hello I am Omar")
print(ans)
print(label_pipeline("1"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


def collate_batch(batch):
    label_list = []
    text_list = []
    offsets = [0]

    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(texto_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))

    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)


train_iter = DBpedia(split="train")
dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)

print(dataloader)

from torch import nn
import torch.nn.functional as F


class ModeloClasificacionTexto(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(ModeloClasificacionTexto, self).__init__()

        # Capa de incrustación (embedding)
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)

        # Capa de normalización por lotes (batch normalization)
        self.bn1 = nn.BatchNorm1d(embed_dim)

        # Capa completamente conectada (fully connected)
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, text, offsets):
        # Incrustar el texto (embed the text)
        embedded = self.embedding(text, offsets)

        # Aplicar la normalización por lotes (apply batch normalization)
        embedded_norm = self.bn1(embedded)

        # Aplicar la función de activación ReLU (apply the ReLU activation function)
        embedded_activated = F.relu(embedded_norm)

        # Devolver las probabilidades de clase (output the class probabilities)
        return self.fc(embedded_activated)


train_iter = DBpedia(split="train")
num_class = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)
embedding_size = 100

modelo = ModeloClasificacionTexto(vocab_size=vocab_size, embed_dim=embedding_size, num_class=num_class).to(device)
print(vocab_size)

# arquitectura
print(modelo)


# Número de parámetros entrenables en nuestro modelo
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"El modelo tiene {count_parameters(modelo):,} parámetros entrenables")


def entrena(dataloader):
    # Colocar el modelo en formato de entrenamiento
    modelo.train()

    # Inicializa accuracy, count y loss para cada epoch
    epoch_acc = 0
    epoch_loss = 0
    total_count = 0

    for idx, (label, text, offsets) in enumerate(dataloader):
        # reestablece los gradientes después de cada batch
        optimizer.zero_grad()
        # Obten predicciones del modelo
        prediccion = modelo(text, offsets)

        # Obten la pérdida
        loss = criterio(prediccion, label)

        # backpropage la pérdida y calcular los gradientes
        loss.backward()

        # Obten la accuracy
        acc = (prediccion.argmax(1) == label).sum()

        # Evita que los gradientes sean demasiado grandes
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 0.1)

        # Actualiza los pesos
        optimizer.step()

        # Llevamos el conteo de la pérdida y el accuracy para esta epoch
        epoch_acc += acc.item()
        epoch_loss += loss.item()
        total_count += label.size(0)

        if idx % 500 == 0 and idx > 0:
            print(
                f" epoca {epoch} | {idx}/{len(dataloader)} batches | perdida {epoch_loss / total_count} | accuracy {epoch_acc / total_count}")

    return epoch_acc / total_count, epoch_loss / total_count


def evalua(dataloader):
    modelo.eval()
    epoch_acc = 0
    total_count = 0
    epoch_loss = 0

    with torch.inference_mode():
        for idx, (label, text, offsets) in enumerate(dataloader):
            # Obtenemos la la etiqueta predecida
            prediccion = modelo(text, offsets)

            # Obtenemos pérdida y accuracy
            loss = criterio(prediccion, label)
            acc = (prediccion.argmax(1) == label).sum()

            # Llevamos el conteo de la pérdida y el accuracy para esta epoch
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            total_count += label.size(0)

    return epoch_acc / total_count, epoch_loss / total_count


# Hiperparámetros

EPOCHS = 1  # epochs
TASA_APRENDIZAJE = 0.2  # tasa de aprendizaje
BATCH_TAMANO = 64  # tamaño de los batches

# Pérdida, optimizador
criterio = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(modelo.parameters(), lr=TASA_APRENDIZAJE)

from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset

# Obten el trainset y testset
train_iter, test_iter = DBpedia()
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)

# Entrenamos el modelo con el 95% de los datos del trainset
num_train = int(len(train_dataset) * 0.95)

# Creamos un dataset de validación con el 5% del trainset
split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

# Creamos dataloaders listos para ingresar a nuestro modelo
train_dataloader = DataLoader(split_train_, batch_size=BATCH_TAMANO, shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_TAMANO, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_TAMANO, shuffle=True, collate_fn=collate_batch)

print("*" * 64)
print("3.8")
print("*" * 64)

# Obten la mejor pérdida
major_loss_validation = float('inf')

# Entrenamos
for epoch in range(1, EPOCHS + 1):
    # Entrenamiento
    entrenamiento_acc, entrenamiento_loss = entrena(train_dataloader)

    # Validación
    validacion_acc, validacion_loss = evalua(valid_dataloader)

    # Guarda el mejor modelo
    if validacion_loss < major_loss_validation:
        best_valid_loss = validacion_loss
        torch.save(modelo.state_dict(), "mejores_guardados.pt")

test_acc, test_loss = evalua(test_dataloader)

print(f'Accuracy del test dataset -> {test_acc}')
print(f'Pérdida del test dataset -> {test_loss}')

print("*" * 64)
print("3.9")
print("*" * 64)

DBpedia_label = {1: 'Company',
                 2: 'EducationalInstitution',
                 3: 'Artist',
                 4: 'Athlete',
                 5: 'OfficeHolder',
                 6: 'MeanOfTransportation',
                 7: 'Building',
                 8: 'NaturalPlace',
                 9: 'Village',
                 10: 'Animal',
                 11: 'Plant',
                 12: 'Album',
                 13: 'Film',
                 14: 'WrittenWork'}


def predict(text, texto_pipeline):
    with torch.no_grad():
        text = torch.tensor(texto_pipeline(text))
        opt_mod = torch.compile(model, mode="reduce-overhead", backend='inductor')
        output = opt_mod(text, torch.tensor([0]))
        return output.argmax(1).item() + 1


ejemplo_1 = "Nithari is a village in the western part of the state of Uttar Pradesh India bordering on New Delhi. " \
            "Nithari forms part of the New Okhla Industrial Development Authority's planned industrial city Noida " \
            "falling in Sector 31. Nithari made international news headlines in December 2006 when the skeletons of a " \
            "number of apparently murdered women and children were unearthed in the village."

model = modelo.to("cpu")

print(f"El ejemplo 1 es de categoría {DBpedia_label[predict(ejemplo_1, texto_pipeline)]}")

print("*" * 64)
print("3.10")
print("*" * 64)
model_state_dict = model.state_dict()
optimizer_state_dict = optimizer.state_dict()

checkpoint = {
    "model_state_dict": model_state_dict,
    "optimizer_state_dict": optimizer_state_dict,
    "epoch": epoch,
    "loss": entrenamiento_loss,
}

torch.save(checkpoint, "model_checkpoint.pth")

print("*" * 64)
print("3.12")
print("*" * 64)
checkpoint = torch.load("model_checkpoint.pth")

train_iter = DBpedia(split="train")
num_class = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)
embedding_size = 100

modelo_2 = ModeloClasificacionTexto(vocab_size=vocab_size, embed_dim=embedding_size, num_class=num_class)
optimizer_2 = torch.optim.SGD(modelo_2.parameters(), lr=0.2)
modelo_2.load_state_dict(checkpoint["model_state_dict"])
optimizer_2.load_state_dict(checkpoint["optimizer_state_dict"])
epoch_2 = checkpoint["epoch"]
loss_2 = checkpoint["loss"]

ejemplo_2 = "Axolotls are members of the tiger salamander, or Ambystoma tigrinum, species complex, along with all " \
            "other Mexican species of Ambystoma."

model_cpu = modelo_2.to("cpu")

ans = DBpedia_label[predict(modelo_2, ejemplo_2, texto_pipeline)]

print(ans)
