import streamlit as st
from main import DocLoader, Embedder, Retriever, AnswerGenerator
import os
from dotenv import load_dotenv

# Load Gemini API key
load_dotenv()
API_KEY = os.getenv("API_KEY")

st.set_page_config(page_title="Chat with your PDF", layout="wide")

# Sidebar - Upload
with st.sidebar:
    st.title("📄 PDF Chatbot")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if st.button("🔁 Reset Chat & Document"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.markdown("---")
    st.markdown("Built with 💡 Streamlit + BERT + Gemini")

# Initialize chatbot state
if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file and "doc_loaded" not in st.session_state:
    # Save uploaded file
    temp_path = "temp.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    with st.spinner("Processing Document..."):
        # Load & index document
        st.session_state.embedder = Embedder()
        st.session_state.retriever = Retriever(st.session_state.embedder)

        doc = DocLoader(temp_path)
        doc.preprocess()
        chunks, page_num = doc.split()

        st.session_state.chunks = chunks
        st.session_state.page_num = page_num
        st.session_state.retriever.build_index(chunks, page_num)

        st.session_state.generator = AnswerGenerator(api_key=API_KEY)
        st.session_state.doc_loaded = True
    st.success("✅ Document processed! Start chatting below.")

# Display chat messages
st.title("💬 Ask Questions About Your PDF")

for chat in st.session_state.messages:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# Only allow chat if doc is loaded
if st.session_state.get("doc_loaded", False):
    prompt = st.chat_input("Ask a question...")
    if prompt:
        # Show user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # RAG Pipeline
        with st.spinner("Thinking..."):
            retrieved_chunks, retrieved_pages = st.session_state.retriever.retrieve(prompt)
            answer = st.session_state.generator.generate(prompt, retrieved_chunks, retrieved_pages)

        if answer:
            st.chat_message("assistant").markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.error("⚠️ Failed to generate a response.")
else:
    st.info("Upload a PDF from the sidebar to get started.")
