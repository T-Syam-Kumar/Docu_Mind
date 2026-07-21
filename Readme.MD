#  DocuMind – PDF RAG Chatbot

An intelligent **Retrieval-Augmented Generation (RAG)** chatbot that allows users to upload PDF documents and ask natural language questions about their content.

Built using **Google Gemini, LangChain, FAISS, HuggingFace Embeddings, and Streamlit**, DocuMind retrieves relevant document chunks and generates accurate answers grounded in the uploaded PDF.

---
##  Live Demo

Try the application online without any installation:

**DocuMind Live:** https://docu-mind-31.streamlit.app/

---

##  Connect With Me

* LinkedIn: https://www.linkedin.com/in/syam-kumar-t-a72508280
* LeetCode: https://leetcode.com/u/TSyamKumar/
* GeeksforGeeks: https://www.geeksforgeeks.org/user/23a31a42j0/

---


##  Features

*  Upload any PDF document
*  Ask questions in natural language
*  Retrieval-Augmented Generation (RAG)
*  Semantic search using vector embeddings
*  Fast document retrieval with FAISS
*  Powered by Gemini 2.5 Flash
*  Adjustable Top-K retrieval results
*  Configurable model temperature
*  Clear chat functionality
*  Modern and responsive Streamlit UI

---

##  Architecture

```text
PDF Upload
    ↓
PDF Parsing (PyPDFLoader)
    ↓
Text Chunking (RecursiveCharacterTextSplitter)
    ↓
Embedding Generation (all-MiniLM-L6-v2)
    ↓
Vector Storage (FAISS)
    ↓
Retriever
    ↓
Gemini 2.5 Flash
    ↓
Answer Generation
```

---

##  Tech Stack

### Frontend

* Streamlit

### LLM

* Gemini 2.5 Flash

### Framework

* LangChain

### Embeddings

* sentence-transformers/all-MiniLM-L6-v2

### Vector Database

* FAISS

### Document Processing

* PyPDFLoader

---

##  Project Structure

```text
DocuMind/
│
├── app.py                     # Streamlit application
├── cli_chatbot.py             # Command line chatbot version
├── requirements.txt
├── .env
│
├── uploads/                   # Uploaded PDFs
│
└── data/
    └── sample.pdf
```

---

##  Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd DocuMind
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux / Mac

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Create a `.env` file

```env
GOOGLE_API_KEY=your_gemini_api_key
```

---

## ▶ Running the Streamlit App

```bash
streamlit run app.py
```

---

##  Running the CLI Version

```bash
python cli_chatbot.py
```

Example:

```text
You : What is the main topic of the document?

AI : The document primarily discusses ...
```

---

##  Future Improvements

* Conversation memory
* Multi-PDF support
* Citation highlighting
* Source chunk visualization
* Chat export functionality
* OCR support for scanned PDFs
* Hybrid retrieval
* Reranking models
* Docker deployment

---

##  Why DocuMind?

Traditional chatbots hallucinate information. DocuMind solves this problem by retrieving relevant information directly from the uploaded document before generating answers, ensuring responses remain grounded in the source material.

---

##  Use Cases

* Research papers
* Academic notes
* Company reports
* Legal documents
* Technical documentation
* Books and manuals
* Resume analysis
* Policy documents

---

##  Author

**Syam Kumar**

B.Tech CSE (AI & ML)

Passionate about Artificial Intelligence, NLP, RAG Systems, and Building Production-Ready AI Applications.

---

##  Support

If you found this project useful, consider giving it a **star** on GitHub and sharing it with others.

Your support motivates further development and improvements.
