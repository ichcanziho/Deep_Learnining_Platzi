from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import bigrams, ngrams
import matplotlib.pyplot as plt
from collections import Counter
from nltk.book import text1
import string
import re


def filter_corpus(corpus, sw, lem):
    filtered_corpus = [word.lower() for word in corpus if word.lower() not in sw]
    filtered_corpus = [word for word in filtered_corpus if word not in string.punctuation]
    filtered_corpus = [word for word in filtered_corpus if len(word) > 5]
    filtered_corpus = [word for word in filtered_corpus if not re.match(r'\d', word)]
    filtered_corpus = [lem.lemmatize(word) for word in filtered_corpus]
    return filtered_corpus


def plot_freq(elementos, figname):
    palabras = [elemento[0] for elemento in elementos]
    frecuencias = [elemento[1] for elemento in elementos]
    plt.plot(palabras, frecuencias)
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.xlabel('Palabras')
    plt.ylabel('Frecuencia')
    plt.title('Frecuencia de palabras')
    plt.tight_layout()
    plt.savefig(f"{figname}.png")
    plt.close()


if __name__ == '__main__':
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    text1_filtered = filter_corpus(text1, stop_words, lemmatizer)
    md_bigrams = list(bigrams(text1_filtered))
    print(md_bigrams[:10])

    bigrams_counter = Counter(md_bigrams)
    mc_bigrams = bigrams_counter.most_common(20)
    # aquí simplemente convertimos una lista de palabras en un texto que contenga todas sus palabras
    mc_bigrams = [(" ".join(grams), freq) for grams, freq in mc_bigrams]
    print(mc_bigrams)
    plot_freq(mc_bigrams, "i3")

    # aquí he decido NO limpiar el text1 para observar las diferencias :)
    md_trigrams = list(ngrams(text1, 3))
    print(md_trigrams[:10])

    trigrams_counter = Counter(md_trigrams)
    mc_trigrams = trigrams_counter.most_common(20)
    mc_trigrams = [(" ".join(grams), freq) for grams, freq in mc_trigrams]
    print(mc_trigrams)
    plot_freq(mc_trigrams, "i4")
