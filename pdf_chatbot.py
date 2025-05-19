import streamlit as st
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import os
import google.generativeai as genai
import time

# 🔐 Load Gemini API Key from Streamlit secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# ✅ Set page config
st.set_page_config(page_title="Chat with your PDF", page_icon="📄")

# Add a professional color scheme to the entire page
st.markdown(
    """
    <style>
    /* General Page Background and Text */
    .stApp {
        background-color: #E0E0E0; /* Light Gray Background */
        color: #000000; /* Black Text */
    }
    .st-emotion-cache-16tyu1 h1 {
    font-size: 2.35rem;
    font-weight: 700;
    padding: 1.25rem 0px 1rem;
}

    /* Title */
    h1 {
        color: #FFD700; /* Golden Title */
        font-weight: bold;
        animation: fadeIn 1.5s ease-in-out;
    }

    /* Subtitle Styling */
    h2 {
        color: #FFD700; /* Golden Subtitle */
        font-size: 20px;
        font-weight: normal;
        margin-top: -15px;
    }

    /* Button Styling */
    .stButton>button {
        background-color: #FFD700; /* Golden Button */
        color: #000000; /* Black Text on Button */
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        transition: background-color 0.3s ease, transform 0.2s ease;
    }

    /* Button Hover Animation */
    .stButton>button:hover {
        background-color: #C0C0C0; /* Silver on Hover */
        cursor: pointer;
        transform: scale(1.05); /* Slight scale-up effect */
    }

    /* Chat Input Field Styling */
    .stTextInput>div>input {
        background-color: #FFFFFF; /* White Background for Input */
        color: #000000; /* Black Text */
        border-radius: 8px;
        border: 2px solid #C0C0C0; /* Silver Border */
        padding: 12px 18px;
        font-size: 16px;
        transition: all 0.3s ease;
    }

    /* Focus Animation for Chat Input */
    .stTextInput>div>input:focus {
        outline: none;
        border-color: #FFD700; /* Golden Border on Focus */
        box-shadow: 0px 0px 6px rgba(255, 215, 0, 0.4); /* Glowing Golden Effect */
    }

    /* Fade-in Animation for Text and Chat History */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Chat History Animation */
    .stChatMessage {
        animation: fadeIn 0.8s ease-in-out;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# Title and Subtitle
st.title("📄 Chat with Your PDF using Gemini")
st.subheader("PDF CHATS MADE BY SAJIDHASAN")

# ✅ Session State Initialization
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "process_complete" not in st.session_state:
    st.session_state.process_complete = False

# ✅ List Available Models from Google
def list_available_models():
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    try:
        models = genai.list_models()
        available_models = [model.name for model in models]
        return available_models
    except Exception as e:
        st.error(f"❌ Error while fetching models: {e}")
        return []

# ✅ PDF Text Extraction
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text

# ✅ Chunk Text for Embedding
def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)

# ✅ Create Conversational Retrieval Chain
def get_conversation_chain(vectorstore):
    # Using free-tier Gemini model
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.7)

    template = """You are a helpful AI assistant that helps users understand their PDF documents.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know — don't make anything up.

{context}

Question: {question}
Helpful Answer:"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

# ✅ Process PDF Uploads
def process_docs(pdf_docs):
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)

    st.session_state.conversation = get_conversation_chain(vectorstore)
    st.session_state.process_complete = True

# ✅ Sidebar — Upload PDFs
with st.sidebar:
    st.subheader("📄 Upload your PDF(s)")
    pdf_docs = st.file_uploader("Select PDF file(s)", type="pdf", accept_multiple_files=True)

    if st.button("Process PDFs") and pdf_docs:
        with st.spinner("🔄 Processing..."):
            try:
                process_docs(pdf_docs)
                st.success("✅ Processing complete!")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ✅ Chat Interface
if st.session_state.process_complete:
    user_question = st.chat_input("Ask a question about your PDF...")

    if user_question:
        with st.spinner("💬 Thinking..."):
            try:
                response = st.session_state.conversation({"question": user_question})
                answer = response["answer"]

                st.session_state.chat_history.append(("user", user_question))
                st.session_state.chat_history.append(("assistant", answer))
            except Exception as e:
                # Handling API quota errors (429)
                if '429' in str(e):
                    st.error("❌ You have exceeded your quota for API calls. Please try again later.")
                else:
                    st.error(f"❌ Chat error: {e}")

    # ✅ Show Chat History
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)
else:
    st.info("👈 Upload your PDF(s) to get started.")
