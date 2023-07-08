from conllu import parse_incr
import numpy as np
import itertools


def get_dicts(data):
    tagCountDict = {}
    emissionDict = {}
    transitionDict = {}

    for sentence in parse_incr(data):
        prev_tag = None
        for word in sentence:
            # Esto representa la etiqueta asociada a la palabra
            current_tag = word['upos']
            # Si la etiqueta no existe aun en el diccionario de etiquetas
            if current_tag not in tagCountDict:
                # Quiere decir que es la primera vez que la vemos, entonces le ponemos un contador de 1
                tagCountDict[current_tag] = 1
            else:
                # Si ya existía significa que entonces al menos tiene un 1 y puedo actualizar su valor con += 1
                tagCountDict[current_tag] += 1

            # Hasta aquí ya he construido mi diccionario que cuenta cuantas veces ha aparecido cada TAG

            # Ahora vamos a repetir el proceso pero para las probabilidades de emision

            # Recordemos que la emision es: Dada una palabra que probabilidad tiene de que tenga cierto tag
            current_word = word['form'].lower()

            # De forma simple es adjuntar la etiqueta de la palabra a la palabra
            emission = current_word + "|" + current_tag
            # Esto como en un ejemplo pasado podría devolver "gobernante|NOUN"

            # Esta técnica de contador es la misma que en el ejemplo anterior
            if emission not in emissionDict:
                emissionDict[emission] = 1
            else:
                emissionDict[emission] += 1

            # Con esto ya tenemos el diccionario que cuenta cuantas veces se ha asignado una etiqueta a una palabra.

            # Ahora vamos a continuar calculando la probabilidad de transición.

            # Recordemos que la probabilidad de transición es: Dada una etiqueta que probablilidad tiene de que tenga otra eitiqueta seguida.

            # Esto básicamente quiere decir, si yo soy NOUN cuál es la etiqueta más probable que pueda tener a continuación.

            # Específicamente para este diccionario se necesita comparar la etiqueta actual y la anterior, entonces debo pasar
            # al menos una iteración para que en la segunda vuelta ya exista mi primer etiqueta anterior
            if prev_tag is None:
                # si la etiqueta anterior no ha sido asignada significa que estoy en la primera vuelta
                # por ende la etiqueta anterior será la actual
                prev_tag = current_tag
                # y utilizo continue para saltarme la siguiente parte de código y continuar con la iteración
                continue

            # si llegamos a esta parte es porque forzosamente estamos al menos en la segunda palabra, lo cual significa
            # que ya existe una prev_tag, por ende ya podemos hacer la transition.
            transition = current_tag + "|" + prev_tag

            # Misma técnica del contador en diccionario
            if transition not in transitionDict:
                transitionDict[transition] = 1
            else:
                transitionDict[transition] += 1
            # Como ya hemos terminado un ciclo ahora podemos decir que la etiqueta anterior fue la actual del inicio
            prev_tag = current_tag
    # Regresamos todos nuestros contadores
    return tagCountDict, emissionDict, transitionDict


def get_matrices(tagCountDict, emissionDict, transitionDict):
    transitionProbDict = {}  # matriz A
    emissionProbDict = {}  # matriz B

    # Recordemos que transition es TAG | TAG
    for key in transitionDict.keys():
        # Cada Key de mi transitionDict tiene 2 elementos que puedo separar por |
        tag, prevtag = key.split('|')
        if tagCountDict[prevtag] > 0:
            # esta es la formula que básicamente es la cantidad de aparición de Ambos TAG juntos entre la cantidad de aparición del TAG anterior
            transitionProbDict[key] = transitionDict[key] / (tagCountDict[prevtag])

    # La lógica en la matriz de emisión es muy similar
    for key in emissionDict.keys():
        word, tag = key.split('|')
        if emissionDict[key] > 0:
            # La fórmula es: la cantidad de apariciones de una palabra dado un tag entre las apariciones de dicho Tag
            emissionProbDict[key] = emissionDict[key] / tagCountDict[tag]
    # Regresamos las matrices
    return transitionProbDict, emissionProbDict


def save_matrices(transitionProbDict, emissionProbDict):
    np.save('outputs/transitionHMM.npy', transitionProbDict)
    np.save('outputs/emissionHMM.npy', emissionProbDict)


def load_matrices(transition_file, emission_file):
    a = np.load(transition_file, allow_pickle='TRUE').item()
    b = np.load(emission_file, allow_pickle='TRUE').item()
    return a, b


if __name__ == '__main__':
    # Leemos los datos de entrada
    data_file = open("../data/UD_Spanish-AnCora/es_ancora-ud-dev.conllu", "r", encoding="utf-8")
    # Obtenemos los contadores necesarios
    tags, emissions, transitions = get_dicts(data_file)
    # Por fines estéticos solo imprimimos parte de los diccionarios.
    print(tags)
    print(dict(itertools.islice(emissions.items(), 5)))
    print(dict(itertools.islice(transitions.items(), 5)))
    # Dado los contadores, podemos obtener las matrices de transición y emisión
    transition_matrix, emission_matrix = get_matrices(tags, emissions, transitions)
    print(dict(itertools.islice(transition_matrix.items(), 5)))
    print(dict(itertools.islice(emission_matrix.items(), 5)))
    # Utilizamos numpy para guardar las matrices en formato npu
    save_matrices(transition_matrix, emission_matrix)
    # Cargamos las matrices que habiamos guardado
    transition_load, emission_load = load_matrices('outputs/transitionHMM.npy', 'outputs/emissionHMM.npy')
    # Predecimos por ejemplo la probabilidad de la transición ADJ dado ADJ
    prob = transition_load['ADJ|ADJ']
    print(prob)
