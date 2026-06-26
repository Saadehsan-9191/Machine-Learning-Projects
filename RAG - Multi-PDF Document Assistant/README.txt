Project: Multi-PDF RAG Document Assistant
Description:
This application allows users to upload multiple PDF files and ask questions based on the combined content of all documents. It uses HuggingFace embeddings, Gemini LLM, FAISS vector search, and Streamlit for the user interface. The system follows a Retrieval-Augmented Generation (RAG) workflow to provide accurate, context-aware answers.

Features:

Upload any number of PDF files

Extract text from all PDFs

Split text into 300-character chunks with 50-character overlap

Convert chunks into embeddings using HuggingFace MiniLM

Store embeddings in a FAISS vector database

Retrieve the top 4 most relevant chunks for each query

Generate answers using Gemini 2.5 Flash

Simple and clean Streamlit interface

Technologies Used:

Python 3

Streamlit

HuggingFace Embeddings (sentence-transformers/all-MiniLM-L6-v2)

Google Gemini via LangChain

FAISS Vector Database

PyPDFLoader

RecursiveCharacterTextSplitter

Installation Steps:

Install Python 3.x

Install required packages:
pip install streamlit
pip install python-dotenv
pip install langchain
pip install langchain-community
pip install langchain-google-genai
pip install langchain-huggingface
pip install langchain-text-splitters
pip install faiss-cpu
pip install pypdf

Create a .env file in the project folder and add your Gemini API key:
GOOGLE_API_KEY=your_key_here

How to Run the App:

Save the main script as app.py

Open a terminal in the project folder

Run the command:
streamlit run app.py

The app will open in your browser at http://localhost:8501

How the System Works (RAG Pipeline):

User uploads one or more PDF files

Each PDF is saved temporarily so PyPDFLoader can read it

Text is extracted and converted into Document objects

Documents are split into overlapping chunks

Each chunk is embedded using HuggingFace MiniLM

Embeddings are stored in a FAISS vector database

When the user asks a question, the system retrieves the top 4 relevant chunks

Gemini generates a final answer using the retrieved context