
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.core import StorageContext, VectorStoreIndex, download_loader

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine


from llama_index.core import Settings

load_dotenv()


DATA_URL = "https://www.gettingstarted.ai/how-to-use-gemini-pro-api-llamaindex-pinecone-index-to-build-rag-app/"

# set llm as Gemini Pro
llm = Gemini(api_key=os.environ["GOOGLE_API_KEY"])
embed_model = GeminiEmbedding(model_name="models/embedding-001")

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512

# create pinecone client
pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# list pinecone indexes
# for index in pinecone_client.list_indexes():
#     print(index['name'])


loader = BeautifulSoupWebReader()
documents = loader.load_data(urls=[DATA_URL])

pine_index = pinecone_client.Index("prodigal")

# Create a PineconeVectorStore using the specified pinecone_index
vector_store = PineconeVectorStore(pinecone_index=pine_index)



pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=1024, chunk_overlap=20),
        embed_model
    ],
    vector_store=vector_store
)

#pipeline.run(documents=documents)

index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
retriever = VectorIndexRetriever(index=index,similarity_top_k=5) 
query_engine = RetrieverQueryEngine(retriever=retriever)


response = query_engine.query("Why should you choose LlamaIndex over other search engines? based on this context")

print(response)
