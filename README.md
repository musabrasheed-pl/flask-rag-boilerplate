# RAG Application with Flask API

This repository contains a Flask application that implements a Retrieval-Augmented Generation (RAG) pipeline. The app allows users to upload PDF documents, process them into embeddings, and ask questions based on the document content. It is designed to be modular, letting you easily switch between different vector stores (Pinecone, FAISS, or Chroma), embedding models, and LLM providers (OpenAI, Claude, or Gemini) via environment variables.

## Features

- **Document Processing:** Extracts text from PDFs and splits it into manageable chunks using LangChain's `RecursiveCharacterTextSplitter`.
- **Vectorstore Integration:** Supports multiple vector stores:
  - **Pinecone**
  - **FAISS**
  - **Chroma**
- **Configurable LLM Integration:** Easily choose among different LLM providers:
  - **OpenAI**
  - **Claude**
  - **Gemini**
- **REST API Endpoints:** Provides API endpoints for processing files and answering queries.

## Prerequisites

- Python 3.8 or higher
- API keys for:
  - **OpenAI**
  - **Pinecone** (if using Pinecone)
  - **Claude** (if using Claude)
  - **Gemini** (if using Gemini)

## Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/musabrasheed-pl/flask-rag-boilerplate.git
   cd flask-rag-boilerplate

2. **Create a Virtual Environment & Install Dependencies**
    
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt

3. **Configure Environment Variables**

    ```bash
   # API Keys
    OPENAI_API_KEY=your_openai_api_key
    PINECONE_API_KEY=your_pinecone_api_key
    CLAUDE_API_KEY=your_claude_api_key      # Only required if using Claude
    GEMINI_API_KEY=your_gemini_api_key        # Only required if using Gemini
    
    # Configuration
    VECTORSTORE_TYPE=chroma   # Options: pinecone, faiss, chroma
    EMBEDDING_MODEL=text-embedding-3-large
    LLM_PROVIDER=openai       # Options: openai, claude, gemini
    LLM_MODEL=gpt-4o          # Or the appropriate model name for your provider

4. **Start the Streamlit App**
    
   ```bash
   python app.py


## Usage
- Process Files Endpoint:
Use the /process_files endpoint to upload one or more PDF files. This endpoint processes the files and initializes the vector store. 
Example using curl:
    ```bash
     curl -X POST -F "files=@/path/to/file.pdf" http://localhost:5000/process_files
    
    
- Ask a Question Endpoint:
Use the /ask endpoint to send a JSON payload containing your query and receive an answer.
Example using curl:
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"query": "What was discussed in the meeting?"}' http://localhost:5000/ask
