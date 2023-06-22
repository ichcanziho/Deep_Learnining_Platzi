from nltk.book import text1
from nltk.corpus import stopwords
from nltk import FreqDist

# Vocabularios: palabras únicas en un corpus
vocab = sorted(set(text1))

print(vocab[:10])

# Distribuciones: frecuencia de aparición
word_freq = FreqDist(text1)
print(word_freq)

print(stopwords.words('spanish')[:10])


def stopwords_percentage(text):
    """
    aqui usamos un recurso léxico (stopwords) para filtrar un corpus
    """
    stopwd = stopwords.words('english')
    content = [w for w in text if w.lower() in stopwd]
    return len(content) / len(text)


swp = stopwords_percentage(text1)
print(swp)
