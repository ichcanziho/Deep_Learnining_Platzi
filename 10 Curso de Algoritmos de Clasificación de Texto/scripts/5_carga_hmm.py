# Para empezar, vamos a cargar nuestras matrices de transición y emisión pre-entrenadas
import numpy as np

transitionProbdict = np.load('outputs/transitionHMM.npy', allow_pickle='TRUE').item()
emissionProbdict = np.load('outputs/emissionHMM.npy', allow_pickle='TRUE').item()

# Vamos a necesitar conocer, y tener enumeradas las diferentes etiquetas (TAG) que puede tener cada palabra.
# Empecemos creando un set de las etiquetas únicas

stateSet = set([w.split('|')[1] for w in list(emissionProbdict.keys())])
print(stateSet)

# Ahora vamos a asignarles un ID a cada TAG (esto nos servirá más adelante para identificar con mayor facilidad a los
# tag como columnas de un frame)

tagStateDict = {state: i for i, state in enumerate(stateSet)}
print(tagStateDict)

# Distribución inicial de estados latentes.

# En este punto lo que nos interesa comprobar es, cuál es la probabilidad de que un TAG sea el inicial de una oración.
# Para resolver esta tarea es relativamente sencillo. Con en dataset Ancora vamos a recorrer cada oración del mismo.
# Apuntamos a la primera palabra y obtenemos su tag (UPOS) creamos un diccionario que cuente la frecuencia de a cada TAG
# cuantas veces fue asignado como inicio de oración. Finalmente, como es una probabilidad, debemos dividir entre el
# total de apariciones, que corresponde con el total de oraciones disponibles en AnCora.

from conllu import parse_incr

data_file = open("../data/UD_Spanish-AnCora/es_ancora-ud-dev.conllu", "r", encoding="utf-8")
data = parse_incr(data_file)
len_sentences = 0
initTagStateProb = {}  # \rho_i^{(0)}
# primero creamos el contador
for token_list in data:
    len_sentences += 1
    tag = token_list[0]['upos']
    if tag in initTagStateProb.keys():
        initTagStateProb[tag] += 1
    else:
        initTagStateProb[tag] = 1
# Ahora noramlizamos dividiendo entre el total de oraciones
for key in initTagStateProb.keys():
    initTagStateProb[key] /= len_sentences

print(initTagStateProb)

# Finalmente, vamos a corroborar que la suma de probabilidades de cada etiqueta sea 1
print(sum(initTagStateProb.values()))


def save_matrices(initTagStateProb, tagStateDict):
    np.save('outputs/initTagStateProbHMM.npy', initTagStateProb)
    np.save('outputs/tagStateDictHMM.npy', tagStateDict)


save_matrices(initTagStateProb, tagStateDict)
