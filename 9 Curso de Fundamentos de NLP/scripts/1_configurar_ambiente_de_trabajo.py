import nltk

# Descargamos un corpus en español
nltk.download('cess_esp')

# Cargamos las oraciones en la variable corpus
corpus = nltk.corpus.cess_esp.sents()
print(corpus)
print(len(corpus))

# Convertimos nuestro arreglo bidimensional en unidimensional
flatten = [item for sublist in corpus for item in sublist]
print(flatten[0:20])
print(len(flatten))
