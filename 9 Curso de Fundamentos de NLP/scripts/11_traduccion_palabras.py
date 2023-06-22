from nltk.corpus import swadesh

# idiomas disponibles
print(swadesh.fileids())

print(swadesh.words('en'))

fr2es = swadesh.entries(['fr', 'es'])
print(fr2es)

translate = dict(fr2es)
print(translate['chien'])

print(translate['jeter'])

