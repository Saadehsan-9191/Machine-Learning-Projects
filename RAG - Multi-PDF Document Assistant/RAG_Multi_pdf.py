# %% [markdown]
# ### LangChain - ChatBot

# %% [markdown]
# 1. Import libraries and load environment variables

# %%
# !pip install streamlit
# !pip install python-dotenv
# !pip install langchain
# !pip install langchain-community
# !pip install langchain-openai
# !pip install langchain-text-splitters
# !pip install faiss-cpu
# !pip install pypdf

import os                                                               # used to access environment variables
import streamlit as st                                                  # streamlit UI framework
import tempfile                                                         # used to create temporary files

from dotenv import load_dotenv                                          # loads .env file
from langchain_community.document_loaders import PyPDFLoader            # loads PDF files
from langchain_text_splitters import RecursiveCharacterTextSplitter     # splits text into chunks
from langchain_openai import OpenAIEmbeddings, ChatOpenAI               # embeddings + LLM
from langchain_community.vectorstores.faiss import FAISS                # FAISS vector database
from langchain_classic.chains import RetrievalQA                        # RAG chain builder
from langchain_huggingface import HuggingFaceEmbeddings                 # HuggingFace embeddings model
from langchain_google_genai import ChatGoogleGenerativeAI               # Google Gemini 2.5 Flash LLM

load_dotenv()                                                           # Load environment variables from .env file




# %% [markdown]
# 2. Initialize Embeddings and LLM

# %%
embeddings = HuggingFaceEmbeddings(                                     # HuggingFace embeddings model
    model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",                                           # Generative AI model - Gemini 2.5 Flash
    temperature=0
)

# %% [markdown]
# 3. Set Up Streamlit UI

# %%
st.title("Multi‑PDF RAG Assistant")

uploaded_files = st.file_uploader(                                      # upload multiple PDFs
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# %% [markdown]
# 4. Extract Text from Uploaded PDF

# %%
all_docs = []                                                           # list to store all PDF pages
if uploaded_files:
    for file in uploaded_files:
                                                                        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            temp_path = tmp.name                                        # this is a real file path

        loader = PyPDFLoader(temp_path)                                 # create loader
        docs = loader.load()                                            # load PDF pages
        all_docs.extend(docs)                                           # add to master list

# %% [markdown]
# 5. Split Text into Chunks

# %%
splitter = RecursiveCharacterTextSplitter(                              # create splitter
    chunk_size=300,                                                     # chunk size
    chunk_overlap=50                                                    # overlap
)

chunks = splitter.split_documents(all_docs)                             # split all PDFs

# %% [markdown]
# 6. Create FAISS Vector Store

# %%
vectorstore = FAISS.from_documents(chunks, embeddings)                  # create FAISS vectorstore from chunks and embeddings

# %% [markdown]
# 8. Create Retriever

# %%
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})            # create retriever with top 4 results

# %% [markdown]
# 9. Create RAG Chain

# %%
qa_chain = RetrievalQA.from_chain_type(                                 # create RAG chain
    llm=llm,                                                            # use the LLM defined earlier
    chain_type="stuff",                                                 # use the "stuff" chain type for combining retrieved documents
    retriever=retriever)                                                # use the retriever defined earlier

# %% [markdown]
# 10. User Question Input and Display Answer

# %%
question = st.text_input("Ask a question about your PDF's")             # user input for question

if question:                                                            # check if question is not empty
    answer = qa_chain.run(question)                                     # run the RAG chain to get the answer
    st.write("### Answer")                                              # display answer
    st.write(answer)                                                    # write the answer to the streamlit app


