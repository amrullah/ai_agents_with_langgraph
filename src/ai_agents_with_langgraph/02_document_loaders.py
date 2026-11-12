import os
from pathlib import Path
import tempfile
import traceback

from langchain_community.document_loaders import (
    TextLoader,
    WebBaseLoader,
    DirectoryLoader,
    PyPDFLoader,
)
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader


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



def web_loader_new_way():
    try:
        loader = DoclingLoader(
            file_path="https://en.wikipedia.org/wiki/Web_scraping",
            export_type=ExportType.MARKDOWN
        )

        # Load the page content into LangChain Documents
        docs = loader.load()

        # Inspect the loaded web document
        print(f"Loaded {len(docs)} document(s).")
        print("\n--- Document Metadata ---")
        print(docs[0].metadata)

        print("\n--- First 100 characters of Markdown page content ---")
        print(docs[0].page_content[:100])

    except Exception as e:
        traceback.print_exc()
        

def document_loader_with_lazy_loading():
    """ No new replacement exists currently for DirectoryLoader, at the time of writing this"""
    # Create temp directory with sample files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample files
        for i in range(5):
            path = Path(tmpdir) / f"doc_{i}.txt"
            path.write_text(f"This is document {i}. It contains sample content.")

        loader = DirectoryLoader(tmpdir, glob="*.txt", loader_cls=TextLoader)

        print("Initialized lazy loader for directory:", tmpdir)
        for doc in loader.lazy_load():
            print("Document Content Preview:", doc.page_content[:50], "...")
            print("Metadata:", doc.metadata["source"])


def pdf_loader_old_way():
    loader = PyPDFLoader("./docs/langchain_demo.pdf")
    documents = loader.load()

    print(f"Loaded {len(documents)} document(s) from PDF")
    for i, doc in enumerate(documents):
        print(f"Document {i+1} Content Preview: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")


def pdf_loader_one_new_way():
    # https://github.com/opendataloader-project/langchain-opendataloader-pdf
    loader = OpenDataLoaderPDFLoader("./docs/langchain_demo.pdf", format='text')
    documents = loader.load()

    print(f"Loaded {len(documents)} document(s) from PDF")
    for i, doc in enumerate(documents):
        print(f"\n\nDocument {i+1} Content Preview: {doc.page_content[:100]}...")
        print(f"\nMetadata: {doc.metadata}")


if __name__ == '__main__':
    # uncomment one at a time:
    # load_text_file_old_way()
    # load_text_file_new_way()
    # web_loader_old_way()
    # web_loader_new_way()
    # document_loader_with_lazy_loading()
    # pdf_loader_old_way()
    pdf_loader_one_new_way()