from langchain.text_splitter import RecursiveCharacterTextSplitter
from hashira.utils import DocsJSONLLoader, get_file_path


def load_documents(file_path: str):
    loader = DocsJSONLLoader(file_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1600, length_function=len, chunk_overlap=160
    )

    return text_splitter.split_documents(data)


def main():

    ans = get_file_path()
    print(ans)
    documents = load_documents(ans)
    print(len(documents))
    print(documents[0])


if __name__ == '__main__':
    main()
