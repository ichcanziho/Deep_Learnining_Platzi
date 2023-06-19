import nltk
nltk.download('book')
from nltk.book import *

# escogemos text1 que es el famoso libro Moby Dick
print(text1)

# Vemos que el texto ya viene tokenizado incluyendo caracteres especiales ....
print(text1.tokens[:10])

# ¿Cuantos tokens tiene el libro?
print(len(text1))

# Primero realizamos la construcción de un vocabulario (identificamos las palabras unicas que hay en el libro)
vocabulario = sorted(set(text1))
print(len(vocabulario))
print(vocabulario[1000:1050])

# luego definimos la medida de riqueza léxica:
rl = len(set(text1)) / len(text1)
print(rl)


# podemos definir funciones en python para estas medidas léxicas:
def riqueza_lexica(texto):
    return len(set(texto)) / len(texto)


def porcentaje_palabra(palabra, texto):
    c = texto.count(palabra)
    return 100 * c / len(texto), c


print(riqueza_lexica(text1))
print(porcentaje_palabra("monster", text1))
