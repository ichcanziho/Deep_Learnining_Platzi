from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
import nltk
nltk.download('cess_esp')
from nltk.book import text1
from pprint import pprint

bigram_measures = BigramAssocMeasures()  # Creamos un objeto bigram_measures que tengas la métrica de PMI
finder = BigramCollocationFinder.from_words(text1)  # Empezamos a buscar los bigramas del corpus text1
finder.apply_freq_filter(20)  # nos quedamos con aquellos que tengan al menos una frecuencia de 20
ans = finder.nbest(bigram_measures.pmi, 10)  # obtenemos los 10 mejores ejemplos usando como base el PMI
pprint(ans)

corpus = nltk.corpus.cess_esp.sents()
print(corpus[:2])
flatten_corpus = [w for l in corpus for w in l]  # Descomprimimos la lista de listas en una sola lista
print(flatten_corpus[:50])

# repetimos el procedimiento anterior pero con el corpus en español
finder = BigramCollocationFinder.from_documents(corpus)
finder.apply_freq_filter(10)
ans = finder.nbest(bigram_measures.pmi, 10)
pprint(ans)
