# streamlit_app.py
import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core import PromptTemplate
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
import chromadb
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_chatbot():
    """Initialize the chatbot with the latest OpenAI models"""
    try:
        load_dotenv()
        
        llm = Gemini(api_key=os.environ["GOOGLE_API_KEY"],model="models/gemini-pro")
        embed_model = GeminiEmbedding(model_name="models/embedding-001")
        
        Settings.llm = llm
        Settings.embed_model = embed_model
        
        load_client = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = load_client.get_collection("constitution")
        
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store)
        
        return index
    
    except Exception as e:
        logger.error(f"Error initializing chatbot: {str(e)}")
        raise

def create_query_engine(index):
    template = (
        """ You are an assistant for question-answering tasks.
        Use the following context to answer the question.
        If you don't know the answer, just say that you don't know. Always give long, detailed answers based on context.\n
        Question: {query_str} \nContext: {context_str} \nAnswer:"""
    )
    qa_prompt = PromptTemplate(template)
    
    return index.as_query_engine(
        text_qa_template=qa_prompt,
        similarity_top_k=7,
        response_mode="compact"
    )

def get_response_text(response):
    """Extract just the response text from the LlamaIndex response object"""
    print(response.response)
    return str(response.response)

def main():
    st.set_page_config(
        page_title="Knowledge AI Agent",
        page_icon="🏪",
        layout="wide"
    )
    
    st.title("Knowledge AI Agent")
    
    # Initialize system
    try:
        if 'query_engine' not in st.session_state:
            with st.spinner("Initializing system..."):
                index = initialize_chatbot()
                st.session_state.query_engine = create_query_engine(index)
            st.success("System initialized successfully!")
        

        
        # Main chat interface
        for message in st.session_state.chat_history:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                st.chat_message("user").write(content)
            else:
                st.chat_message("assistant").markdown(content)
        
        # Chat input
        if query := st.chat_input("Ask about products..."):
            st.chat_message("user").write(query)
            
            try:
                with st.spinner("Analyzing your question..."):
                    response = st.session_state.query_engine.query(query)
                    response_text = get_response_text(response)
                
                # Display the response
                st.chat_message("assistant").markdown(response_text)
                
                # Update chat history
                st.session_state.chat_history.extend([
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": response_text}
                ])
                
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
                st.info("Please try rephrasing your question.")
    
    except Exception as e:
        st.error(f"System Error: {str(e)}")
        st.warning("Please check your configuration and try again.")

if __name__ == "__main__":
    main()