# Importamos el módulo nltk y descargamos el tokenizador punkt para utilizar la función word_tokenize más adelante.
import numpy as np
from nltk import word_tokenize
from pprint import pprint


# Definición de la función ViterbiMatrix que implementa el algoritmo de Viterbi
def viterbi_matrix(secuencia, transitionProbdict, emissionProbdict,
                  tagStateDict, initTagStateProb):
    # Tokenizamos la secuencia de entrada utilizando word_tokenize y almacenamos los tokens en la variable seq.
    seq = word_tokenize(secuencia)
    # Creamos una matriz viterbiProb de dimensiones (17, len(seq)) para almacenar las probabilidades del algoritmo de
    # Viterbi. El número 17 representa las categorías posibles para las etiquetas (upos).
    viterbiProb = np.zeros((len(tagStateDict), len(seq)))  # upos tiene 17 categorias

    # Inicializamos la primera columna de la matriz viterbiProb. Para cada etiqueta en tagStateDict, calculamos la
    # etiqueta de la palabra (word_tag) y verificamos si existe una probabilidad de emisión para esa etiqueta de palabra
    # en emissionProbdict. Si existe, multiplicamos la probabilidad inicial de la etiqueta (initTagStateProb[key]) por
    # la probabilidad de emisión y la asignamos a viterbiProb[tag_row, 0].
    for key in tagStateDict.keys():
        tag_row = tagStateDict[key]
        word_tag = seq[0].lower() + '|' + key
        if word_tag in emissionProbdict.keys():
            viterbiProb[tag_row, 0] = initTagStateProb[key] * emissionProbdict[word_tag]

    # Calculamos las siguientes columnas de la matriz viterbiProb. Para cada columna (col) a partir de la segunda
    # columna, y para cada etiqueta en tagStateDict, calculamos la etiqueta de la palabra (word_tag) y verificamos si
    # existe una probabilidad de emisión para esa etiqueta de palabra en emissionProbdict. Si existe, recorremos todas
    # las etiquetas en tagStateDict para obtener las probabilidades posibles en la columna anterior. Si hay una
    # probabilidad mayor a cero en la columna anterior (viterbiProb[tag_row2, col-1] > 0), multiplicamos esa
    # probabilidad por la probabilidad de transición y la probabilidad de emisión, y las agregamos a la lista
    # possible_probs. Finalmente, asignamos a viterbiProb[tag_row, col] el máximo valor de possible_probs.
    for col in range(1, len(seq)):
        for key in tagStateDict.keys():
            tag_row = tagStateDict[key]
            word_tag = seq[col].lower() + '|' + key
            if word_tag in emissionProbdict.keys():
                # Miramos los estados de la columna anterior
                possible_probs = []
                for key2 in tagStateDict.keys():
                    tag_row2 = tagStateDict[key2]
                    tag_prevtag = key + '|' + key2
                    if tag_prevtag in transitionProbdict.keys():
                        if viterbiProb[tag_row2, col - 1] > 0:
                            possible_probs.append(
                                viterbiProb[tag_row2, col - 1] * transitionProbdict[tag_prevtag] * emissionProbdict[
                                    word_tag])
                viterbiProb[tag_row, col] = max(possible_probs)
    # Devolvemos la matriz viterbiProb.
    return viterbiProb, seq


def get_viterbi_tags(seq, transitionProbdict, emissionProbdict,
                  tagStateDict, initTagStateProb):

    viterbi_prob, seq = viterbi_matrix(seq, transitionProbdict, emissionProbdict,
                  tagStateDict, initTagStateProb)
    res = []
    # enumeramos cada palabra de la oración y recorremos todas ellas
    for i, p in enumerate(seq):
        # empezamos a recorrer todos los tags disponibles
        for tag in tagStateDict.keys():
            # si la probabilidad actual de este tag es ARGMAX de la columna,
            if tagStateDict[tag] == np.argmax(viterbi_prob[:, i]):
                # entonces en el resultado, adjunto la palabra y el tag
                res.append((p, tag))

    return res


def load_matrices(transition_file, emission_file, tag_state_file, init_tag_state_file):
    a = np.load(transition_file, allow_pickle='TRUE').item()
    b = np.load(emission_file, allow_pickle='TRUE').item()
    c = np.load(tag_state_file, allow_pickle='TRUE').item()
    d = np.load(init_tag_state_file, allow_pickle='TRUE').item()
    return a, b, c, d


# Llamada a la función ViterbiMatrix con una secuencia de entrada
if __name__ == '__main__':
    # Cargamos los datos necesarios
    a = 'outputs/transitionHMM.npy'
    b = 'outputs/emissionHMM.npy'
    c = 'outputs/tagStateDictHMM.npy'
    d = 'outputs/initTagStateProbHMM.npy'
    transitionProbdict, emissionProbdict, tagStateDict, initTagStateProb = load_matrices(a, b, c, d)
    # Definimos la función ViterbiMatrix que implementa el algoritmo de Viterbi. Los parámetros transitionProbdict,
    # emissionProbdict, tagStateDict e initTagStateProb son diccionarios que contienen las probabilidades de
    # transición, las probabilidades de emisión, los estados de etiquetas y las probabilidades iniciales de los
    # estados de etiquetas, respectivamente.

    # matrix = viterbi_matrix('el mundo es ', transitionProbdict, emissionProbdict, tagStateDict, initTagStateProb)
    # print(matrix)
    msg = "el mundo es muy pequeño"
    response = get_viterbi_tags(msg, transitionProbdict, emissionProbdict, tagStateDict, initTagStateProb)
    pprint(response)
    print()
    msg = "estos instrumentos han de rasgar"
    response = get_viterbi_tags(msg, transitionProbdict, emissionProbdict, tagStateDict, initTagStateProb)
    pprint(response)
