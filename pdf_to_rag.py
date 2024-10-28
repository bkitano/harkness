import os
from typing import List
from pymupdf4llm import to_markdown
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.schema import Document


def convert_pdfs_to_markdown(*, pdf_directory: str, output_directory: str) -> List[str]:
    """
    Convert all PDFs in a directory to markdown files.
    """
    markdown_files = []
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, filename)
            markdown_text = to_markdown(pdf_path)

            markdown_filename = os.path.splitext(filename)[0] + ".md"
            markdown_path = os.path.join(output_directory, markdown_filename)

            with open(markdown_path, "w", encoding="utf-8") as f:
                f.write(markdown_text)

            markdown_files.append(markdown_path)

    return markdown_files


def load_and_process_documents(*, markdown_files: List[str]) -> List[Document]:
    """
    Load markdown files and convert them to LangChain Document objects.
    """
    documents = []
    for file_path in markdown_files:
        loader = TextLoader(file_path)
        documents.extend(loader.load())
    return documents


def chunk_documents(*, documents: List[Document]) -> List[Document]:
    """
    Chunk the documents into smaller pieces.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_documents(documents)


def create_vectorstore(*, chunks: List[Document], vectordb_directory: str) -> Chroma:
    """
    Create a vector store from the document chunks.
    """
    if not os.path.exists(vectordb_directory):
        os.makedirs(vectordb_directory)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        chunks, embeddings, persist_directory=vectordb_directory
    )
    return vectorstore


def pdf_to_rag(
    *, pdf_directory: str, output_directory: str, vectordb_directory: str
) -> Chroma:
    """
    Main function to convert PDFs to a RAG-ready vector store.
    """
    # if the directory doesn't exist, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Convert PDFs to markdown
    markdown_files = convert_pdfs_to_markdown(
        pdf_directory=pdf_directory, output_directory=output_directory
    )

    # Load and process documents
    documents = load_and_process_documents(markdown_files=markdown_files)

    # Chunk documents
    chunks = chunk_documents(documents=documents)

    # Create vector store
    vectorstore = create_vectorstore(
        chunks=chunks, vectordb_directory=vectordb_directory
    )

    return vectorstore


if __name__ == "__main__":
    pdf_dir = "downloaded_works/William Merrill"
    output_dir = "output_markdown"
    vectordb_dir = "vectordb"

    vectorstore = pdf_to_rag(
        pdf_directory=pdf_dir,
        output_directory=output_dir,
        vectordb_directory=vectordb_dir
    )
    print(f"Vector store created with {vectorstore._collection.count()} chunks.")
