from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from collections import Counter
from nltk.book import text1
import string
import re


def clean_corpus(corpus, sw, lem, min_words, most_common):
    """
    Limpia un corpus de texto aplicando diferentes filtros.

    Args:
        corpus (list): Corpus de texto a limpiar.
        sw (set): Conjunto de palabras vacías (stop words) a eliminar.
        lem (WordNetLemmatizer): Objeto lematizador de palabras.
        min_words (int): Mínimo número de palabras en una palabra filtrada.
        most_common (int): Número de palabras más comunes a extraer.

    Returns:
        tuple: Una tupla que contiene el diccionario de palabras filtradas y una lista de las palabras más comunes.
    """
    # empezamos minimizando todas las palabras y eliminando las palabras que sean stopwords
    filtered_corpus = [word.lower() for word in corpus if word.lower() not in sw]
    # eliminamos signos de puntación
    filtered_corpus = [word for word in filtered_corpus if word not in string.punctuation]
    # eliminamos palabras que tengan una longitud menor o igual a 5 letras
    filtered_corpus = [word for word in filtered_corpus if len(word) > 5]
    # eliminamos letras que no sean texto
    filtered_corpus = [word for word in filtered_corpus if not re.match(r'\d', word)]
    # lematizamos cada palabra
    filtered_corpus = [lem.lemmatize(word) for word in filtered_corpus]
    # obtenemos un contador de las palabras en el corpus
    filtered_corpus = Counter(filtered_corpus)
    # guardamos las palabras más comunes
    mc = filtered_corpus.most_common(most_common)
    # eliminamos palabras cuya frecuencia sea menor a un umbral
    elementos_filtrados = {clave: valor for clave, valor in filtered_corpus.items() if valor > min_words}
    # regresamos el contador y las palabras más comunes
    return elementos_filtrados, mc


def plot_freq(elementos):
    """
       Grafica la frecuencia de las palabras en un diccionario.

       Args:
           elementos (list): Lista de tuplas que contienen palabras y sus frecuencias.
       """
    palabras = [elemento[0] for elemento in elementos]
    frecuencias = [elemento[1] for elemento in elementos]
    plt.plot(palabras, frecuencias)
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.xlabel('Palabras')
    plt.ylabel('Frecuencia')
    plt.title('Frecuencia de palabras')
    plt.tight_layout()
    plt.savefig("i2.png")


if __name__ == '__main__':

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    vocabulario_filtrado, vocabulario_popular = clean_corpus(text1, stop_words, lemmatizer, 5, 20)
    print(vocabulario_filtrado)
    print(vocabulario_popular)
    plot_freq(vocabulario_popular)

