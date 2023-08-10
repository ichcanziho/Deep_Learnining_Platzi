from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("./public_key_cryptography.pdf")
data = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    length_function=len,
    chunk_overlap=200
)

documents = text_splitter.split_documents(data)

print(len(documents))

print(documents[20])

print("*"*64)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap  = 50,
    length_function = len,
)

# o

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 1000,
#     chunk_overlap  = 100,
#     length_function = len,
# )

documents = text_splitter.split_documents(data)
print("tamaño:", len(documents))
print("tipo:", type(documents))
print("tipo doc", type(documents[0]))
print("Contenido pag 0:")
print(documents[0].page_content)
