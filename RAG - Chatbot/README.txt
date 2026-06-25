PDF Question Answering Chatbot (LangChain + Gemini + HuggingFace)

This project uses:
- HuggingFace embeddings (all-MiniLM-L6-v2)
- Gemini 2.5 Flash LLM
- FAISS vector store
- Streamlit UI
- pdfplumber for PDF text extraction

Workflow:
1. User uploads a PDF.
2. Text is extracted using pdfplumber.
3. Text is split into chunks using RecursiveCharacterTextSplitter.
4. Chunks are embedded using HuggingFace embeddings.
5. FAISS builds a vector store from the embeddings.
6. Retriever fetches the top relevant chunks.
7. Gemini LLM generates the final answer.
8. Streamlit displays the answer.

Environment:
- GOOGLE_API_KEY stored in .env


Notes:
- HuggingFace embeddings are used because Gemini embeddings currently cause 404 errors in LangChain.
- Gemini LLM works perfectly for answering questions.
- This setup is fully free and cloud-based.