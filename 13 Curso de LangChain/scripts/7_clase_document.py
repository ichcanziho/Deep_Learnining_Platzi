from langchain.schema import Document

page_content = "Textooooooooolargoooooo ejemplo"
metadata = {'fuente': 'platzi', 'clase': 'langchain'}

doc = Document(
    page_content=page_content, metadata=metadata
)

print(doc.page_content)
print(doc)
