import os
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.readers.file import PDFReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
load_dotenv()

PDF_PATH = "constitution.pdf"  # Path to the PDF file

# Set up system prompt for ingestion
system_prompt = "You are a helpful assistant tasked with embedding and storing key information from PDFs. Focus on preserving key factual details during the embedding process."

# Set LLM as Gemini Pro with system prompt
llm = Gemini(api_key=os.environ["GOOGLE_API_KEY"], system_prompt=system_prompt)
embed_model = GeminiEmbedding(model_name="models/embedding-001")

# Create Pinecone client
pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Load data from PDF
pdf_reader = PDFReader()
documents_pdf = pdf_reader.load_data(file=PDF_PATH)

# Create a Pinecone index
pine_index = pinecone_client.Index("prodigal")

# Create a PineconeVectorStore
vector_store = PineconeVectorStore(pinecone_index=pine_index)

# Ingestion Pipeline to process and store the PDF data
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=1024, chunk_overlap=20),
        embed_model
    ],
    vector_store=vector_store
)

# Run the pipeline to ingest the PDF data
pipeline.run(documents=documents_pdf)
