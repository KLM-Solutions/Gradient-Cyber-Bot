import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager
from pinecone import Pinecone
from PyPDF2 import PdfReader
import os
from langsmith import Client
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "gradient_cyber_customer_bot"

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# Ensure all required API keys are present
if not all([OPENAI_API_KEY, PINECONE_API_KEY, LANGCHAIN_API_KEY]):
    st.error("Missing one or more required API keys. Please check your .env file or Streamlit secrets.")
    st.stop()

# Initialize Pinecone and OpenAI
INDEX_NAME = "gradientcyber"

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "gradientcyber"

# Initialize LangChain components
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = PineconeVectorStore(pinecone_api_key=PINECONE_API_KEY, index_name=index_name, embedding=embeddings)

# Initialize LangSmith client
client = Client(api_key=LANGCHAIN_API_KEY)

# Initialize LangChain tracer and callback manager
tracer = LangChainTracer(project_name="gradient_cyber_customer_bot", client=client)
callback_manager = CallbackManager([tracer])
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-4o",
    temperature=0,
    callback_manager=callback_manager
)

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + " "
    return text

def process_and_upsert_pdf(pdf_file):
    text = extract_text_from_pdf(pdf_file)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    # Generate a unique ID for this document
    doc_id = str(uuid.uuid4())
    
    # Add the doc_id to the metadata of each chunk
    metadatas = [{"source": pdf_file.name, "doc_id": doc_id} for _ in chunks]
    
    vectorstore.add_texts(chunks, metadatas=metadatas)
    
    # Store the doc_id in session state
    if "doc_ids" not in st.session_state:
        st.session_state.doc_ids = []
    st.session_state.doc_ids.append(doc_id)
    
    return len(chunks)

# Streamlit UI
st.sidebar.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
        color: #008080;
    }
    </style>
    <p class="big-font">Gradient Cyber</p>
    """, unsafe_allow_html=True)
st.sidebar.title("PDF Uploader")

uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    st.sidebar.write("File uploaded successfully!")
    if st.sidebar.button("Process and Upsert to Pinecone"):
        with st.sidebar.spinner("Processing PDF and upserting to Pinecone..."):
            num_chunks = process_and_upsert_pdf(uploaded_file)
            st.sidebar.success(f"Processed and upserted {num_chunks} chunks to Pinecone.")

st.title("Gradient Cyber Q&A System")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Q&A section
query = st.chat_input("Ask a question about the uploaded documents:")
if query:
    st.session_state.messages.append({"role": "human", "content": query})
    with st.chat_message("human"):
        st.markdown(query)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Create a ConversationalRetrievalChain with a filtered retriever
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Create a filtered retriever
        retriever = vectorstore.as_retriever()
        
        # Filter the retriever to only search documents with the stored doc_ids
        if "doc_ids" in st.session_state and st.session_state.doc_ids:
            retriever = vectorstore.as_retriever(
                search_kwargs={
                    "filter": {"doc_id": {"$in": st.session_state.doc_ids}}
                }
            )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            callback_manager=callback_manager
        )
        
        # Generate the response
        result = qa_chain({"question": query, "chat_history": [(msg["role"], msg["content"]) for msg in st.session_state.messages]})
        full_response = result["answer"]
        message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})

st.sidebar.title("Conversation History")
if st.sidebar.button("Clear History"):
    st.session_state.messages = []
    if "doc_ids" in st.session_state:
        st.session_state.doc_ids = []
    st.sidebar.success("Conversation history and document references cleared.")

st.write("Note: Make sure you have set up your Pinecone index and OpenAI API key correctly.")
