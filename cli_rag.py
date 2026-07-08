import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

load_dotenv()


loader = PyPDFLoader("data/sample.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = splitter.split_documents(documents)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


db = FAISS.from_documents(
    docs,
    embeddings
)

retriever = db.as_retriever(
    search_kwargs={"k": 3}
)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant.

Answer the user's question ONLY from the provided context and remove if any extra special characters.


If the answer is not found in the context,
reply:

"I couldn't find that information in the document."

Context:
{context}
"""
        ),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(
    llm,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)


print("\n==============================")
print("      PDF RAG Chatbot")
print("==============================")
print("Type 'exit' to quit.\n")

while True:

    query = input("You : ")

    if query.lower() == "exit":
        break

    response = rag_chain.invoke(
        {
            "input": query
        }
    )

    print("\nAI :", response["answer"])
    print("-" * 60)
