import gdown

file_url = 'https://drive.google.com/uc?id=1kihb-PiE0jLnlJicZ42yDCIpTo_D40Zc'
output_file = 'data/repos_cairo.csv'
gdown.download(file_url, output_file, quiet=False)
print(f"Archivo descargado como '{output_file}'")

import pandas as pd

df = pd.read_csv('data/repos_cairo.csv')
print(df.head())

from langchain.document_loaders import DataFrameLoader

loader = DataFrameLoader(df, page_content_column="repo_name")
data = loader.load()

print(f"El archivo es de tipo {type(data)} y tiene una longitud de {len(data)} debido a la cantidad de observaciones en el CSV.")

from pprint import pprint

pprint(data[:5])
