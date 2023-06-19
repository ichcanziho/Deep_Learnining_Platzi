import re
import nltk

nltk.download('wordnet')
print("="*64)
nltk.download('punkt')
print("="*64)


texto = """ Cuando sea el rey del mundo  (imaginaba él en su cabeza) no tendré que  preocuparme por estas bobadas.
            Era solo un niño de 7 años, pero pensaba que podría ser cualquier cosa que su imaginación le permitiera visualizar en su cabeza ..."""

print(texto)

# Caso 1: tokenizacion más simple: por espacios vacios !
print(re.split(r' ', texto))

# Caso 2: tokenización usando expresiones regulares
print(re.split(r'[ \s]+', texto))

# RegEx reference: \W -> all characters other than letters, digits or underscore
print(re.split(r'[ \W\s]+', texto))

# nuestra antigua regex no funciona en este caso:
texto = 'En los E.U. esa postal vale $15.50 ...'

print(re.split(r'[ \W\s]+', texto))

pattern = r'''(?x)                 # set flag to allow verbose regexps
              (?:[A-Z]\.)+         # abbreviations, e.g. U.S.A.
              | \w+(?:-\w+)*       # words with optional internal hyphens
              | \$?\d+(?:\.\d+)?%? # currency and percentages, e.g. $12.40, 82%
              | \.\.\.             # ellipsis
              | [][.,;"'?():-_`]   # these are separate tokens; includes ], [
'''

print(nltk.regexp_tokenize(texto, pattern))

from nltk import word_tokenize

print(word_tokenize(texto, language="spanish"))

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

words = ["running", "runs", "ran"]

for word in words:
    stemmed_word = stemmer.stem(word)
    print(word, "->", stemmed_word)


from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

words = ["running", "runs", "ran"]

for word in words:
    lemmatized_word = lemmatizer.lemmatize(word)
    print(word, "->", lemmatized_word)
