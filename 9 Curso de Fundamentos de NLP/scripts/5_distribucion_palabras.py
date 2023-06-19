import nltk
nltk.download('book')
from nltk.book import text1
from nltk import FreqDist
from time import time
import matplotlib.pyplot as plt

start = time()
# METODO NO recomendable para conjuntos muy grandes
dic = {}
for palabra in set(text1):
    # dic[palabra] = porcentaje_palabra(palabra, text1)
    dic[palabra] = text1.count(palabra)

print("Execution's time:", time()-start, "seconds")
print(dic)
start = time()
# NLTK tiene un metodo muy eficiente
fdist = FreqDist(text1)
print("Execution's time:", time()-start, "seconds")
print(fdist.most_common(20))
fdist.plot(20)
plt.close()
