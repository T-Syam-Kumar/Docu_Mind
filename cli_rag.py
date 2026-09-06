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


from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
)

from datasets import Dataset

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

Answer the user's question ONLY from the provided context.

Do not use outside knowledge.

Do not add unnecessary special characters.

If the answer is not found in the context, reply:

I couldn't find that information in the document. Please ask only from the document.

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




def evaluate_response(question, answer, context):

    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [context]
    }

    dataset = Dataset.from_dict(data)

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy
        ]
    )

    return result




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


    answer = response["answer"]



    context = []

    for document in response["context"]:
        context.append(document.page_content)



    print("\nAI :", answer)



    try:

        evaluation = evaluate_response(
            query,
            answer,
            context
        )

        print("\n---------- RAGAS EVALUATION ----------")

        print(
            "Faithfulness :",
            evaluation["faithfulness"]
        )

        print(
            "Answer Relevancy :",
            evaluation["answer_relevancy"]
        )

        print("--------------------------------------")


    except Exception as e:

        print("\nEvaluation Error:", e)


    print("-" * 65)
