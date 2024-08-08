import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager
from pinecone import Pinecone
import os
from langsmith import Client
from dotenv import load_dotenv

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
index = pc.Index(INDEX_NAME)

# Initialize LangChain components
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = PineconeVectorStore(pinecone_api_key=PINECONE_API_KEY, index_name=INDEX_NAME, embedding=embeddings)

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

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main content area
st.title("Gradient Cyber Q&A System")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask a question about the documents in Pinecone:")

if query:
    st.session_state.messages.append({"role": "human", "content": query})
    with st.chat_message("human"):
        st.markdown(query)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Create a retriever that searches within all documents in the Pinecone index
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Adjust 'k' as needed
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            callback_manager=callback_manager
        )
        
        result = qa_chain({"question": query, "chat_history": [(msg["role"], msg["content"]) for msg in st.session_state.messages]})
        full_response = result["answer"]
        message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})

st.sidebar.title("Gradient Cyber")
if st.sidebar.button("Clear Conversation"):
    st.session_state.messages = []
    st.success("Conversation history cleared.")

st.write("Note: This system is searching all documents stored in your Pinecone index.")
