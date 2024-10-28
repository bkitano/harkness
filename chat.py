import os
from typing import List
from pymupdf4llm import to_markdown
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.schema import Document


def retrieve_docs(*, query: str) -> List[Document]:
    vectordb_dir = "vectordb"

    vectorstore = Chroma(
        persist_directory=vectordb_dir, embedding_function=OpenAIEmbeddings()
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 6}
    )

    retrieved_docs = retriever.invoke(query)

    return retrieved_docs


if __name__ == "__main__":
    print(retrieve_docs(query="What is a limit of transformers?"))
