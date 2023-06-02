import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Comprobar la versión de PyTorch
print(torch.__version__)
# Crea *nuevos* parámetros
volumen = 0.8
sesgo = 0.2

# Crea datos
inicio = 0
final = 1
step = 0.025
X = torch.arange(inicio, final, step).unsqueeze(dim=1)
print(f"Shape de X: {X.shape}")
y = volumen * X + sesgo
print(f"Shape de y: {y.shape}")

print(X[:10], y[:10])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(len(X_train), len(X_test))


def plot_predictions(datos_ent=X_train,
                     etiq_ent=y_train,
                     datos_prueba=X_test,
                     etiq_prueba=y_test,
                     predictions=None,
                     name=""):
    """
    Traza datos de entrenamiento, datos de prueba y compara predicciones
    """
    plt.figure(figsize=(10, 10))

    # Traza datos de entrenamiento en verde
    plt.scatter(datos_ent, etiq_ent, c="b", s=6, label="Datos de entrenamiento")

    # Traza datos de prueba en amarillo
    plt.scatter(datos_prueba, etiq_prueba, c="y", s=6, label="Datos de prueba")

    if predictions is not None:
        # Traza las predicciones en rojo
        plt.scatter(datos_prueba, predictions, c="r", s=6, label="Predicciones")

    # Leyenda
    plt.legend(prop={"size": 12})
    plt.savefig(f"datos_{name}.png")
    plt.close()


plot_predictions(name="1")

print("*" * 64)


# Crea una clase de modelo de regresión lineal
class ModeloRegresionLineal(nn.Module):
    def __init__(self):
        super().__init__()
        self.volumen = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        self.sesgo = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

    # Define el cálculo en el modelo
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.volumen * x + self.sesgo


torch.manual_seed(42)
# Crea un objeto instanciando la clase ModeloRegresionLineal

model_1 = ModeloRegresionLineal()

print(model_1)

print(model_1.state_dict())

# Hacer predicciones con el modelo
with torch.inference_mode():
    y_predc = model_1(X_test)

# Comprueba las predicciones
print(y_predc)

plot_predictions(predictions=y_predc, name="2")

print("*" * 64)

# Crea función de pérdida
fn_perd = torch.nn.L1Loss()

# Crea el optimizador
optimizador = torch.optim.SGD(params=model_1.parameters(), lr=0.01)  # tasa de aprendizaje (cuánto debe cambiar el
# optimizador de parámetros en cada paso, más alto = más (menos estable), más bajo = menos (puede llevar mucho tiempo))

print(fn_perd)

torch.manual_seed(42)

# Establezca cuántas veces el modelo pasará por los datos de entrenamiento
epocas = 100

# Cree listas de vacías para realizar un seguimiento de nuestro modelo
entrenamiento_loss = []
test_loss = []

for epoca in range(epocas):
    # Entrenamiento

    # Pon el modelo en modo entrenamiento
    model_1.train()

    # 1. Pase hacia adelante los datos usando el método forward()
    y_predc = model_1(X_train)

    # 2. Calcula la pérdida (Cuán diferentes son las predicciones de nuestros modelos)
    perdida = fn_perd(y_predc, y_train)

    # 3. Gradiente cero del optimizador
    optimizador.zero_grad()

    # 4. Pérdida al revés
    perdida.backward()

    # 5. Progreso del optimizador
    optimizador.step()

    # Función de prueba

    # Pon el modelo en modo evaluación
    model_1.eval()

    with torch.inference_mode():

        # 1. Reenviar datos de prueba
        prueba_predc = model_1(X_test)

        # 2. Calcular la pérdida en datos de prueba
        prueb_perd = fn_perd(prueba_predc, y_test.type(torch.float))

        # Imprime lo que está pasando
        if epoca % 10 == 0:
            entrenamiento_loss.append(perdida.detach().numpy())
            test_loss.append(prueb_perd.detach().numpy())
            print(f"Epoca: {epoca} | Entrenamiento pérdida: {perdida} | Test pérdida {prueb_perd}")

# Traza las curvas de pérdida
plt.plot(entrenamiento_loss, label="Perd entrenamiento")
plt.plot(test_loss, label="Perd prueba")
plt.ylabel("Pérdida")
plt.xlabel("Epoca")
plt.legend()
plt.savefig(f"datos_perdida.png")
plt.close()
# 1. Configura el modelo en modo de evaluación
model_1.eval()

# 2. Configura el administrador de contexto del modo de inferencia
with torch.inference_mode():
    # 3. Asegúrate de que los cálculos se realicen con el modelo y los datos en el mismo dispositivo en nuestro caso,
    # nuestros datos y modelo están en la CPU de forma predeterminada
    # model_1.to(device)
    # X_prueb = X_prueb.to(device)
    y_predc = model_1(X_test)

print(y_predc)

plot_predictions(predictions=y_predc, name="3")
