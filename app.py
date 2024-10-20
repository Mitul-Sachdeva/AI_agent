
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Settings

load_dotenv()

# Define the system prompt for the LLM
system_prompt = """You are a knowledgeable assistant tasked with answering questions using the most relevant information from the provided dataset."""

# Set up LLM as Gemini Pro with system prompt
llm = Gemini(api_key=os.environ["GOOGLE_API_KEY"], system_prompt=system_prompt)
embed_model = GeminiEmbedding(model_name="models/embedding-001")
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512
# Create Pinecone client
pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Access the Pinecone index
pine_index = pinecone_client.Index("prodigalrag")

# Create a PineconeVectorStore
vector_store = PineconeVectorStore(pinecone_index=pine_index)

# Create VectorStoreIndex from the vector store
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# Set up retriever to get the top 5 similar results
retriever = VectorIndexRetriever(index=index, similarity_top_k=5)

# Create the query engine
query_engine = RetrieverQueryEngine(retriever=retriever)

while(True):
    query = input("Enter your question: ")
    response = query_engine.query(query)
    print(response)
    print("\n")