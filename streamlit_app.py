import streamlit as st
from pinecone import Pinecone
from PyPDF2 import PdfReader
import openai
import io
import time
from collections import deque
from langsmith import Client
import functools
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Pinecone and OpenAI
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
LANGCHAIN_API_KEY = st.secrets["LANGCHAIN_API_KEY"]
INDEX_NAME = "gradientcyber"

# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["LANGCHAIN_PROJECT"] = "Gradient-Cyber-QA-System"

# Initialize clients
pc = Pinecone(api_key=PINECONE_API_KEY)
langsmith_client = Client(api_key=LANGCHAIN_API_KEY)
chat = ChatOpenAI(model_name="gpt-4o", temperature=0.3)
embeddings = OpenAIEmbeddings()

# Specify your index name here
index = pc.Index(INDEX_NAME)

# Initialize session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = deque(maxlen=5)  # Keep last 5 Q&A pairs

def safe_run_tree(name, run_type):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                with langsmith_client.trace(name=name, run_type=run_type) as run:
                    result = func(*args, **kwargs)
                    run.end(outputs={"result": str(result)})
                    return result
            except Exception as e:
                st.error(f"Error in LangSmith tracing: {str(e)}")
                return func(*args, **kwargs)
        return wrapper
    return decorator

@safe_run_tree(name="extract_text_from_pdf", run_type="chain")
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + " "
    return text

def create_chunks(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

@safe_run_tree(name="get_embedding", run_type="llm")
def get_embedding(text):
    with get_openai_callback() as cb:
        embedding = embeddings.embed_query(text)
    return embedding

@safe_run_tree(name="upsert_to_pinecone", run_type="chain")
def upsert_to_pinecone(chunks, pdf_name):
    batch_size = 50  # Reduced batch size
    total_chunks = len(chunks)
    progress_bar = st.progress(0)
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i+batch_size]
        ids = [f"{pdf_name}_{j}" for j in range(i, i+len(batch))]
        try:
            # Get embeddings for the batch
            embeddings = [get_embedding(chunk) for chunk in batch]
            # Prepare vectors for upsert
            to_upsert = [
                (id, embedding, {"text": chunk})
                for id, embedding, chunk in zip(ids, embeddings, batch)
            ]
            # Upsert to Pinecone
            index.upsert(vectors=to_upsert)
            # Update progress bar
            progress = min(1.0, (i + batch_size) / total_chunks)
            progress_bar.progress(progress)
        except Exception as e:
            st.error(f"Error during upsert: {str(e)}")
            st.error(f"Failed at chunk {i}")
            break
        # Add a small delay to avoid hitting rate limits
        time.sleep(1)  # Increased delay
    st.success(f"Finished processing {total_chunks} chunks.")

@safe_run_tree(name="search_pinecone", run_type="chain")
def search_pinecone(query, top_k=5):
    query_embedding = get_embedding(query)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return results

@safe_run_tree(name="generate_answer", run_type="llm")
def generate_answer(query, context, conversation_history):
    system_message = SystemMessage(content="You are a helpful assistant that answers questions about customer details based on the provided context. Always strive to give accurate and relevant information. If the question is not relevant to the provided context, clearly state that the question cannot be answered based on the available information.")
    
    messages = [system_message]
    
    # Add conversation history
    for q, a in conversation_history:
        messages.append(HumanMessage(content=q))
        messages.append(SystemMessage(content=a))
    
    # Add current query and context
    messages.append(HumanMessage(content=f"Context: {context}\n\nQuestion: {query}"))
    
    # Make 5 LLM calls
    answers = []
    for _ in range(5):
        with get_openai_callback() as cb:
            response = chat(messages)
        answers.append(response.content.strip())
    
    # Choose the most common answer or the first one if all are different
    from collections import Counter
    most_common_answer = Counter(answers).most_common(1)[0][0]
    return most_common_answer

# Sidebar
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

# File uploader in sidebar
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    st.sidebar.write("File uploaded successfully!")
    if st.sidebar.button("Process and Upsert to Pinecone"):
        with st.sidebar.spinner("Processing PDF and upserting to Pinecone..."):
            # Extract text from the uploaded PDF
            pdf_text = extract_text_from_pdf(io.BytesIO(uploaded_file.read()))
            # Create chunks from the extracted text
            chunks = create_chunks(pdf_text)
            # Upsert chunks to Pinecone
            upsert_to_pinecone(chunks, uploaded_file.name)

st.sidebar.title("Conversation History")
if st.sidebar.button("Show Conversation History"):
    for q, a in st.session_state.conversation_history:
        st.sidebar.text(f"Q: {q}")
        st.sidebar.text(f"A: {a}")
        st.sidebar.write("---")

# Main content area
st.title("Gradient Cyber Q&A System")

# Q&A section
query = st.text_input("Enter your question:")
if st.button("Ask") or query:
    if query:
        with langsmith_client.trace(name="process_query", run_type="chain") as run:
            with st.spinner("Searching for relevant information..."):
                search_results = search_pinecone(query)
                context = " ".join([result['metadata']['text'] for result in search_results['matches']])
            with st.spinner("Generating answer..."):
                answer = generate_answer(query, context, st.session_state.conversation_history)
            if "cannot be answered" in answer.lower() or "not relevant" in answer.lower():
                st.error(answer)
            else:
                st.subheader("Answer:")
                st.write(answer)
            # Add to conversation history
            st.session_state.conversation_history.append((query, answer))
            run.end(outputs={"answer": answer})
    else:
        st.warning("Please enter a question.")

st.write("Note: Make sure you have set up your Pinecone index correctly.")
