from langchain.document_loaders import UnstructuredFileLoader

loader = UnstructuredFileLoader("./public_key_cryptography.pdf")
data = loader.load()

print("tipo:", type(data), "tamaño:", len(data))
print("Ejemplo Metadata")
print(data[0].metadata)
print("Ejemplo Content")
print(data[0].page_content[:300])

from langchain.document_loaders import PyPDFLoader

print("*"*64)
loader = PyPDFLoader("./public_key_cryptography.pdf")
data = loader.load()
print("tipo:", type(data), "tamaño:", len(data))
print("Ejemplo Metadata")
print(data[0].metadata)
print("Ejemplo Content")
print(data[0].page_content[:300])