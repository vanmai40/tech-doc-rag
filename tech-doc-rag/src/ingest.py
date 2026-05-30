from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import FAISS
from pinecone import Pinecone

from config import settings


def get_embeddings():
    if settings.embedding_provider == "huggingface":
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    else:
        return OpenAIEmbeddings(
            base_url=settings.embedding_base_url or settings.llm_base_url,
            api_key=settings.embedding_api_key or settings.llm_api_key,
            model=settings.embedding_model or "nomic-embed-text",
        )


def load_documents(data_dir: str) -> list:
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        show_progress=True,
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from {data_dir}")
    return documents


def chunk_documents(documents: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks


def ingest_to_vectorstore(chunks: list):
    embeddings = get_embeddings()

    if settings.use_local_vectorstore:
        print(f"Building FAISS index with {len(chunks)} chunks...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(settings.faiss_index_path)
        print(f"FAISS index saved to {settings.faiss_index_path}")
    else:
        pc = Pinecone(api_key=settings.pinecone_api_key)
        print(f"Uploading {len(chunks)} chunks to Pinecone index '{settings.pinecone_index_name}'...")
        PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=settings.pinecone_index_name,
        )
        print("Upload complete!")


def main():
    data_dir = Path(__file__).parent.parent / "data"

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("Please create the data directory and add your documents.")
        return

    documents = load_documents(str(data_dir))
    chunks = chunk_documents(documents)
    ingest_to_vectorstore(chunks)

    print("\nIngestion complete! You can now run the agent with: make run")


if __name__ == "__main__":
    main()
