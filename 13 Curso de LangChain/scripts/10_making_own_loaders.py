from langchain.schema import Document
import jsonlines
from typing import List


class TransformerDocsJSONLLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:  # Solo ayuda a decir que la salida es una lista cuyos elementos son Documents
        with jsonlines.open(self.file_path) as reader:
            documents = []
            # Para cada entrada del documento JSONL
            for obj in reader:
                page_content = obj.get("text", "")
                metadata = {
                    'title': obj.get("title", ""),
                    'repo_owner': obj.get("repo_owner", ""),
                    'repo_name': obj.get("repo_name", ""),
                }
                # Añadimos el page content como text y metadata con title, repo_owner y repo_name
                documents.append(
                    Document(page_content=page_content, metadata=metadata)
                )
        return documents


loader = TransformerDocsJSONLLoader("data/transformers_docs.jsonl")
data = loader.load()

for doc in data:
    print(doc)

