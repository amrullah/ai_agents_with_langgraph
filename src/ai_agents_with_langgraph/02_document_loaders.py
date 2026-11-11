import os
import tempfile

from langchain_community.document_loaders import (
    TextLoader,
    WebBaseLoader,
    DirectoryLoader,
    PyPDFLoader,
)
from langchain_docling import DoclingLoader


def load_text_file_old_way():
    """This way is deprecated as langchain_community is not under active maintenance"""
    # Create a temporary text file for demonstration
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(
            b"Hello, this is a sample text file.\nThis file is used to demonstrate the TextLoader."
        )
        temp_file_path = temp_file.name

    try:
        # Load the text file using TextLoader
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        print(f"Loaded {len(documents)} document(s)")
        # print(f"Content preview: {documents[0].page_content[:100]}...")
        print(f"Metadata: {documents[0].metadata}")

        # Print the loaded documents
        for doc in documents:
            print("Document Content:")
            print(doc)
            print(doc.page_content)
    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)


def load_text_file_new_way():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(
            b"Hello, this is a sample text file.\nThis file is used to demonstrate the TextLoader."
        )
        temp_file_path = temp_file.name


    try:
        loader = DoclingLoader(file_path=temp_file_path)

        documents = loader.load()

        print(f"Loaded {len(documents)} document(s)")
        # print(f"Content preview: {documents[0].page_content[:100]}...")
        print(f"Metadata: {documents[0].metadata}")

        # Print the loaded documents
        for doc in documents:
            print("Document Content:")
            print(doc)
            print(doc.page_content)

    finally:
        os.remove(temp_file_path)


def web_loader_old_way():
    loader = WebBaseLoader(
        "https://en.wikipedia.org/wiki/Web_scraping", bs_kwargs={"parse_only": None}
    )
    documents = loader.load()

    print(f"Loaded {len(documents)} document(s) from web")
    print(f"Source: {documents[0].metadata.get('source', 'N/A')}")
    print(f"Content length: {len(documents[0].page_content)} characters")
    print(f"Preview: {documents[0].page_content[:200]}...")


if __name__ == '__main__':
    # uncomment one at a time:
    # load_text_file_old_way()
    # load_text_file_new_way()
    web_loader_old_way()