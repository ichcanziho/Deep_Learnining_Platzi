from nltk.corpus import wordnet as wn


def show_syns(word):
    # Obtener los synsets (conjunto de sinónimos) para la palabra proporcionada
    ss = wn.synsets(word, lang='spa')

    for syn in ss:
        # Imprimir el nombre del synset y su definición
        print(syn.name(), ': ', syn.definition())

        for name in syn.lemma_names():
            # Imprimir cada nombre de lema (sinónimo)
            print(' * ', name)

    return ss


# Obtener los synsets para la palabra 'perro' y mostrarlos
ss = show_syns('perro')

# Obtener los synsets para la palabra 'gato' y mostrarlos
ss2 = show_syns('gato')

# Obtener los synsets para la palabra 'animal' y mostrarlos
ss3 = show_syns('animal')

# Obtener el primer synset de 'perro'
perro = ss[0]

# Obtener el primer synset de 'gato'
gato = ss2[0]

# Obtener el primer synset de 'animal'
animal = ss3[0]

# Calcular y mostrar la similitud de ruta entre 'animal' y 'perro'
print(animal.path_similarity(perro))  # 0.3333333333333333

# Calcular y mostrar la similitud de ruta entre 'animal' y 'gato'
print(animal.path_similarity(gato))  # 0.125

# Calcular y mostrar la similitud de ruta entre 'perro' y 'gato'
print(perro.path_similarity(gato))  # 0.2

# Calcular y mostrar la similitud de ruta entre 'perro' y 'perro' (misma palabra)
print(perro.path_similarity(perro))  # 1.0

