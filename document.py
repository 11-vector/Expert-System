import typing as tp
import itertools as it
import functools as fn
import os
import pathlib
from langchain_docling.loader import DoclingLoader
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DOC_DIR = pathlib.Path("./docs/cleaned/")


def get_raw_files(ext: tp.Literal["pdf", "docx"] | None):
    filenames = list(BASE_DOC_DIR.iterdir())
    if ext is None:
        return filenames
    filtered_files = list(filter(lambda a: a.suffix == f".{ext}", filenames))
    return filtered_files


def load_and_split_document(filename: os.PathLike) -> list[Document]:
    # filename = str(filename)
    ext = os.path.splitext(filename)[-1].lower()
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=500,
        chunk_overlap=10,
        # separators=["\n\n", "\n", " ", ""]
    )
    loader_fn = PyPDFLoader if ext == ".pdf" else DoclingLoader
    loader = loader_fn(filename)
    pages = loader.load()
    doc_splits = text_splitter.split_documents(pages)
    for doc in doc_splits:
        source = doc.metadata.get('source')
        doc.metadata = dict(
            machine_name=source.stem,
        )
    return doc_splits


@fn.lru_cache(maxsize=2)
def load_documents(ext: str | None = None):
    fnames = get_raw_files(ext)
    docs = list(it.chain.from_iterable(load_and_split_document(fname)
                for fname in fnames))
    return docs


def main():
    docs = load_documents()
    print(f"Document counts = {len(docs)}")


if __name__ == '__main__':
    main()
