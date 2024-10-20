import os
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone import ServerlessSpec
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.readers.file import PDFReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
load_dotenv()

PDF_PATH = "constitution.pdf"  # Path to the PDF file

print("Starting the Setup pipeline")
llm = Gemini(api_key=os.environ["GOOGLE_API_KEY"])
embed_model = GeminiEmbedding(model_name="models/embedding-001")

# Create Pinecone client
pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Load data from PDF
pdf_reader = PDFReader()
print(f"Loading data from {PDF_PATH}")
documents_pdf = pdf_reader.load_data(file=PDF_PATH)

# Create a Pinecone index
index_name = 'prodigalrag'
if index_name not in pinecone_client.list_indexes():
    pinecone_client.create_index(name=index_name,dimension=768,metric="cosine",
    spec=ServerlessSpec(
    cloud="aws",
    region="us-east-1"
  ),)
    print(f"Created index {index_name}")
pine_index = pinecone_client.Index(index_name)


# Create a PineconeVectorStore
vector_store = PineconeVectorStore(pinecone_index=pine_index)

# Ingestion Pipeline to process and store the PDF data
print("Running the ingestion pipeline")
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=20),
        embed_model
    ],
    vector_store=vector_store
)

# Run the pipeline to ingest the PDF data
print("Running the pipeline")
pipeline.run(documents=documents_pdf)
