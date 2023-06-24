import nltk
nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn
from pprint import pprint

ss = wn.synsets('carro', lang='spa')
pprint(ss)

# Explorando los synsets
for syn in ss:
    print(syn.name(), ': ', syn.definition())
    for name in syn.lemma_names():
        print(' * ', name)

# Palabras más específicas
pprint(ss[0].hyponyms())

# Palabras más generales
pprint(ss[0].hypernyms())

import networkx as nx
import matplotlib.pyplot as plt


def closure_graph(synset, fn):
    """
       Crea un grafo de cierre a partir de un synset dado y una función.

       Args:
           synset (nltk.corpus.reader.wordnet.Synset): Synset de inicio.
           fn (function): Función que toma un Synset y devuelve una lista de Synsets.

       Returns:
           tuple: Tupla que contiene el grafo y un diccionario de etiquetas.
       """
    seen = set()
    graph = nx.DiGraph()
    labels = {}

    def recurse(s):
        """
                Función recursiva para construir el grafo de cierre.

                Args:
                    s (nltk.corpus.reader.wordnet.Synset): Synset actual.
                """
        if not s in seen:
            seen.add(s)
            labels[s.name] = s.name().split('.')[0]
            graph.add_node(s.name)
            for s1 in fn(s):
                graph.add_node(s1.name)
                graph.add_edge(s.name, s1.name)
                recurse(s1)

    recurse(synset)
    return graph, labels


def draw_text_graph(G, labels, filename):
    """
        Dibuja un grafo de texto utilizando NetworkX y Matplotlib.

        Args:
            G (networkx.DiGraph): Grafo dirigido.
            labels (dict): Diccionario de etiquetas de nodos.
            filename (str): Nombre del archivo a guardar
        """
    plt.figure(figsize=(18, 12))
    pos = nx.planar_layout(G, scale=18)
    nx.draw_networkx_nodes(G, pos, node_color="red", linewidths=0, node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=20, labels=labels)
    nx.draw_networkx_edges(G, pos)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filename)
    plt.close()


# Conceptos que son más específicos que la palabra raíz de la cual derivan.
print(ss[0].name())
G, labels = closure_graph(ss[0], fn=lambda s: s.hyponyms())
draw_text_graph(G, labels, "carro_hypo.png")

# Conceptos que son más generales.
print(ss[0].name())
G, labels = closure_graph(ss[0], fn=lambda s: s.hypernyms())
draw_text_graph(G, labels, "carro_hyper.png")

