from dotenv import load_dotenv
import os

load_dotenv()

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings

if __name__ == "__main__":

    # 1. Loading Documents
    print("Loading Documents...")
    loader = TextLoader("./information.txt")
    documents = loader.load()  # loads the documents with metadata
    print(f"Loaded {len(documents)} documents")

    # 2. Splitting Documents
    print("Splitting Documents...")
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Split each document individually if the loader returned a list
    split_documents = []
    for document in documents:
        split_documents.extend(splitter.split_text(document['text']))  # Assuming document is a dictionary with 'text' key

    print(f"Split {len(documents)} documents into {len(split_documents)} chunks")

    # 3. Embedding Documents
    print("Started Embedding Documents...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Inserting Documents into VectorDB
    print("Inserting Documents into VectorDB...")
    from pinecone import init

    # Initialize Pinecone if not already initialized
    init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-west1-gcp")  # Make sure to set the environment according to your Pinecone instance

    # Insert into Pinecone index
    vector_db = Pinecone.from_documents(split_documents, embeddings, index_name=os.getenv("INDEX_NAME"))
    print(f"Inserted {len(split_documents)} documents into VectorDB")
