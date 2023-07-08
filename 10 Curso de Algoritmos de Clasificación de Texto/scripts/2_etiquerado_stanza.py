# esta parte puede demorar un poco ....
import stanza

stanza.download('es')

nlp = stanza.Pipeline('es', processors='tokenize,pos')
doc = nlp('yo soy una persona muy amable')
print(doc)

for sentence in doc.sentences:
    for word in sentence.words:
        print(word.text, word.pos)
