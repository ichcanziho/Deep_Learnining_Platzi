import numpy as np
from nltk.book import text1
from nltk import bigrams, FreqDist
import pandas as pd
# esta biblioteca me permite hacer gráficos interactivos
import plotly.express as px

md_bigrams = list(bigrams(text1))

threshold = 2
# Obtenemos los bigramas donde cada palabra tenga más de 2 letras
filtered_bigrams = [bigram for bigram in md_bigrams if len(bigram[0]) > threshold and len(bigram[1]) > threshold]
# Usamos un contador para obtener la frecuencia de los bigramas
filtered_bigram_dist = FreqDist(filtered_bigrams)
# Obtenemos las palabras individuales que tengan más de 2 letras
filtered_words = [word for word in text1 if len(word) > threshold]
# Usamos un contador para obtener la frecuencia de las palabras (unigramas)
filtered_word_dist = FreqDist(filtered_words)

# Para entender mejor las variables que vamos a utilizar vamos a construir un DataFrame de pandas
df = pd.DataFrame()
df['bi_gram'] = list(set(filtered_bigrams))
df['word_0'] = df['bi_gram'].apply(lambda x: x[0])
df['word_1'] = df['bi_gram'].apply(lambda x: x[1])
# Aquí estamos usando los diccionarios de filtered_bigram_dist y filtered_word_dist para obtener la frecuencia de una
# palabra con base justamente en esa palabra
df['bi_gram_freq'] = df['bi_gram'].apply(lambda x: filtered_bigram_dist[x])
df['word_0_freq'] = df['word_0'].apply(lambda x: filtered_word_dist[x])
df['word_1_freq'] = df['word_1'].apply(lambda x: filtered_word_dist[x])
print(df)

# Calculamos un pseudo PMI (NO es directamente la probabilidad de las palabras) pero es similar usar
# la frecuencia de las palabras
df['PMI'] = np.log2(df['bi_gram_freq'] / (df['word_0_freq'] * df['word_1_freq']))
df['log(bi_gram_freq)'] = np.log2(df['bi_gram_freq'])
df.sort_values(by='PMI', ascending=False, inplace=True)
print(df)

# Finalmente, hacemos un scatter de la distrbiución de los bigramas
fig = px.scatter(x=df['PMI'].values, y=df['log(bi_gram_freq)'].values, color=df['PMI'] + df['log(bi_gram_freq)'],
                 size=(df['PMI'] + df['log(bi_gram_freq)']).apply(lambda x: 1 / (1 + abs(x))).values,
                 hover_name=df['bi_gram'].values, width=600, height=600,
                 labels={'x': 'PMI', 'y': 'Log(Bigram Frequency)'})
fig.show()
