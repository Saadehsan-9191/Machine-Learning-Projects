# %% [markdown]
# ### LangChain - ChatBot

# %% [markdown]
# 1. Import libraries and load environment variables

# %%
# !pip install python-dotenv streamlit pdfplumber langchain faiss-cpu 
# !pip install langchain-text-splitters
# !pip install --upgrade langchain
# !pip install langchain_community
# !pip install langchain-core
# !pip install langchain-huggingface sentence-transformers
# !pip install langchain-ollama
# !pip install faiss-cpu
# !pip install langchain-google-genai



import os                                                           # Import OS module to interact with environment variables and system functions
from dotenv import load_dotenv                                      # Load environment variables from a .env file
import streamlit as st                                              # Streamlit for building the web app UI
import pdfplumber 
from langchain_text_splitters import RecursiveCharacterTextSplitter # Split large text into smaller chunks
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores.faiss import FAISS            # FAISS vector store from langchain-community package
from langchain_core.documents import Document                       # Document class from langchain-core package
from langchain_classic.chains import RetrievalQA                    # RetrievalQA chain from langchain-classic package

load_dotenv()                                                       # Load environment variables from .env file




# %% [markdown]
# 2. Initialize Embeddings and LLM

# %%
# embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")  # Initialize Google Generative AI embeddings model

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   # free model
    temperature=0
)

# %% [markdown]
# 3. Set Up Streamlit UI

# %%
st.title("PDF Question Answering with LangChain")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# %% [markdown]
# 4. Extract Text from Uploaded PDF

# %%
full_text = ""
if uploaded_file is not None:
    with pdfplumber.open(uploaded_file) as pdf:
        full_text = "".join(page.extract_text() or "" for page in pdf.pages)
if not full_text.strip():
        st.error("❌ No extractable text found. This PDF may be scanned or image-based.")
        st.stop()

# %% [markdown]
# 5. Split Text into Chunks

# %%
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_text(full_text)

# %% [markdown]
# 6. Create Document Objects for Vector Store

# %%
docs = [Document(page_content=chunk) for chunk in chunks]

# %% [markdown]
# 7. Create FAISS Vector Store

# %%
vectorstore = FAISS.from_documents(docs, embeddings)

# %% [markdown]
# 8. Create Retriever

# %%
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# %% [markdown]
# 9. Create RetrievalQA Chain

# %%
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# %% [markdown]
# 10. User Question Input and Display Answer

# %%
question = st.text_input("Ask a question about the PDF")

if question:
    answer = qa_chain.run(question)
    st.write("### Answer")
    st.write(answer)


