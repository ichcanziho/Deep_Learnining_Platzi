import torch
import torch.nn as nn


class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        # Capa de embedding para mapear índices de palabras a vectores de embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Capa LSTM para procesar los embeddings y obtener estados ocultos
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True)

        # Capa lineal para proyectar el último estado oculto en la salida final
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # Obtener los embeddings de las palabras
        embedded = self.embedding(text)

        # Propagar los embeddings a través de la capa LSTM
        output, (hidden, cell) = self.rnn(embedded)

        # Seleccionar el último estado oculto como estado final
        final_hidden = hidden[-1]

        # Pasar el estado final a través de la capa lineal para obtener la salida final
        return self.fc(final_hidden)


# Parámetros del modelo
vocab_size = 10000
embedding_dim = 100
hidden_dim = 256
output_dim = 2

# Crear instancia del modelo
model = TextClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)

# Imprimir la estructura del modelo
print(model)
