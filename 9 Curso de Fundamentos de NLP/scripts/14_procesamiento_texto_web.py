import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk import word_tokenize
from urllib import request

# Texto plano desde web
url = "http://www.gutenberg.org/files/2554/2554-0.txt"
response = request.urlopen(url)
raw = response.read().decode('utf8')

print(raw[:200])


tokens = word_tokenize(raw)
print(tokens[:20])

text = nltk.Text(tokens)
a = text.collocations()
print(a)

# Procesar HTML
import requests
from bs4 import BeautifulSoup
import re
from nltk.tokenize import RegexpTokenizer
url = 'https://www.gutenberg.org/files/2701/2701-h/2701-h.htm'
r = requests.get(url)

html = r.text
print(html[:200])

soup = BeautifulSoup(html, 'html.parser')
# print(soup)

text = soup.get_text()
tokens = re.findall('\w+', text)
print(tokens[:10])

tokenizer = RegexpTokenizer('\w+')
tokens = tokenizer.tokenize(text)
tokens = [token.lower() for token in tokens]
print(tokens[:10])

text = nltk.Text(tokens)
a = text.collocations()
print(a)