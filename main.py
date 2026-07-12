import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()



st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide"
)


st.markdown("""
<style>

.stApp{
    background:#0E1117;
}

.block-container{
    padding-top:2rem;
}

.title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:white;
}

.subtitle{
    text-align:center;
    color:gray;
    margin-bottom:30px;
}

</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------

st.markdown(
    "<div class='title'> PDF RAG Chatbot</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Made using Gemini + LangChain + FAISS</div>",
    unsafe_allow_html=True
)



if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


with st.sidebar:

    st.header("📂 Upload PDF")

    pdf = st.file_uploader(
        "Choose PDF",
        type="pdf"
    )

    st.divider()

    k = st.slider(
        "Top K Results",
        1,
        10,
        3
    )

    temperature = st.slider(
        "Temperature",
        0.0,
        1.0,
        0.0
    )

    st.divider()

    if st.button("🗑 Clear Chat"):

        st.session_state.messages = []

    st.divider()

    st.header("📚 Source Chunks")



if pdf is not None:

    # Build only once for each uploaded PDF
    if st.session_state.pdf_name != pdf.name:

        with st.spinner("Processing PDF..."):

            os.makedirs("uploads", exist_ok=True)

            pdf_path = os.path.join(
                "uploads",
                pdf.name
            )

            with open(pdf_path, "wb") as f:
                f.write(pdf.getbuffer())

            loader = PyPDFLoader(pdf_path)

            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            docs = splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            vector_db = FAISS.from_documents(
                docs,
                embeddings
            )

            retriever = vector_db.as_retriever(
                search_kwargs={
                    "k": k
                }
            )

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=temperature
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
You are an AI assistant.

Answer ONLY using the context.

If the answer is not present in the document,
say:

I couldn't find that information in the document.

Context:

{context}
"""
                    ),
                    (
                        "human",
                        "{input}"
                    )
                ]
            )

            document_chain = create_stuff_documents_chain(
                llm,
                prompt
            )

            rag_chain = create_retrieval_chain(
                retriever,
                document_chain
            )

            st.session_state.rag_chain = rag_chain
            st.session_state.pdf_name = pdf.name

        st.success("✅ PDF processed successfully!")


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])



question = st.chat_input(
    "Ask anything about your PDF..."
)

if question:

    if st.session_state.rag_chain is None:

        st.warning("Please upload a PDF first.")

    else:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    response = st.session_state.rag_chain.invoke(
                        {
                            "input": question
                        }
                    )

                    answer = response["answer"]

                except Exception as e:

                    answer = f"❌ {e}"

                st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )
