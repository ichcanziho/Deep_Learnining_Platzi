from langchain.embeddings import SentenceTransformerEmbeddings

embeddings_st = SentenceTransformerEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Otro modelo en español que podríamos usar es "symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli"

documentos_a_incrustar = [
    "¡Hola parce!",
    "¡Uy, hola!",
    "¿Cómo te llamas?",
    "Mis parceros me dicen Omar",
    "¡Hola Mundo!"
]

incrustaciones = embeddings_st.embed_documents(documentos_a_incrustar)
print(len(incrustaciones))

incrustacion = embeddings_st.embed_query(documentos_a_incrustar[0])
print(len(incrustacion))

from langchain.embeddings import HuggingFaceInstructEmbeddings

# A junio de 2023 no hay modelos Instruct para español
embedding_instruct = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large",
    model_kwargs={"device": "cuda"}
)

# El device podría ser cpu
incrustaciones = embedding_instruct.embed_documents(documentos_a_incrustar)
print(len(incrustaciones[4]))

incrustacion = embedding_instruct.embed_query(documentos_a_incrustar[0])
print(len(incrustacion))

print(embedding_instruct.client, embeddings_st.client)