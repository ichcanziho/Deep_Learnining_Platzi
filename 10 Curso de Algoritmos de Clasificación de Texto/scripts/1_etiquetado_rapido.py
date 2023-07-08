# Dependencias previas
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import word_tokenize
nltk.download('cess_esp')
from nltk.corpus import cess_esp as cess
from nltk import UnigramTagger as ut
from nltk import BigramTagger as bt
# Etiquetado en una línea:

# Etiquetado en una línea
text = word_tokenize("And now here I am enjoying today")
print(nltk.pos_tag(text))

# Categoria gramatical de cada etiqueta
nltk.download('tagsets')
for _, tag in nltk.pos_tag(text):
    print(nltk.help.upenn_tagset(tag))

# Palabras homónimas
text = word_tokenize("They do not permit other people to get residence permit")
print(nltk.pos_tag(text))

# Etiquetado en Español


# Entrenamiendo del tagger por unigramas

cess_sents = cess.tagged_sents()
fraction = int(len(cess_sents)*90/100)
uni_tagger = ut(cess_sents[:fraction])
p = uni_tagger.evaluate(cess_sents[fraction+1:])
print(p)

print(uni_tagger.tag("Yo soy una persona muy amable".split(" ")))

fraction = int(len(cess_sents)*90/100)
bi_tagger = bt(cess_sents[:fraction])
p = bi_tagger.evaluate(cess_sents[fraction+1:])
print(p)

print(bi_tagger.tag("Yo soy una persona muy amable".split(" ")))
