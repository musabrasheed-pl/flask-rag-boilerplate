from flask import Flask, request, jsonify
import os
import time
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import pinecone
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma, Pinecone as LangchainPinecone
from langchain_community.chat_models import ChatOpenAI, ChatGooglePalm, ChatAnthropic
from flask_swagger_ui import get_swaggerui_blueprint

# Load environment variables from the .env file
load_dotenv()

# === Configuration Variables ===
VECTORSTORE_TYPE = os.getenv("VECTORSTORE_TYPE", "pinecone").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Swagger UI setup
SWAGGER_URL = "/docs"  # URL for Swagger UI
API_URL = "/static/swagger.json"  # JSON file with API spec

app = Flask(__name__)
swagger_ui_blueprint = get_swaggerui_blueprint(SWAGGER_URL, API_URL)
app.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)

# Global storage for conversation history, vectorstore, and processed chunks
conversation_history = []
vectorstore_global = None
chunks_global = None

# ====================================
#          RAG Application Class
# ====================================
class RAGApp:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.validate_key(self.openai_api_key, "OPENAI_API_KEY")
        self.embedding = OpenAIEmbeddings(openai_api_key=self.openai_api_key, model=EMBEDDING_MODEL)

    @staticmethod
    def validate_key(key, key_name):
        if not key:
            raise ValueError(f"Please set up your {key_name} in the .env file")

    def process_files(self, files):
        """Process uploaded PDF files and split text into chunks."""
        all_chunks = []
        for file in files:
            if not file.filename.endswith(".pdf"):
                continue
            text = self.extract_text_from_pdf(file)
            chunks = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128).split_text(text)
            all_chunks.extend([(chunk, file.filename) for chunk in chunks])
        return all_chunks

    @staticmethod
    def extract_text_from_pdf(file):
        pdf_reader = PdfReader(file)
        return "".join(page.extract_text() or "" for page in pdf_reader.pages)

    def initialize_vectorstore(self, chunks):
        """Initialize vectorstore based on configuration."""
        if VECTORSTORE_TYPE == "pinecone":
            return self.initialize_pinecone_vectorstore()
        elif VECTORSTORE_TYPE == "faiss":
            texts = [chunk for chunk, _ in chunks]
            return FAISS.from_texts(texts, self.embedding)
        elif VECTORSTORE_TYPE == "chroma":
            texts = [chunk for chunk, _ in chunks]
            return Chroma.from_texts(texts, self.embedding, collection_name="documents")
        else:
            raise ValueError("Unsupported VECTORSTORE_TYPE. Please use 'pinecone', 'faiss', or 'chroma'.")

    def initialize_pinecone_vectorstore(self):
        self.validate_key(os.getenv("PINECONE_API_KEY"), "PINECONE_API_KEY")
        index_name = PINECONE_INDEX_NAME
        dimension = 1580
        pc = pinecone.init(api_key=os.getenv("PINECONE_API_KEY"))
        existing_indexes = pinecone.list_indexes()
        if index_name not in existing_indexes:
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")
            )
            self.wait_for_index(pc, index_name)
        index = pc.Index(index_name)
        return LangchainPinecone(index, self.embedding, "text")

    @staticmethod
    def wait_for_index(pc, index_name):
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    def setup_qa_chain(self, vectorstore):
        """Set up the QA retrieval chain using the configured LLM and vectorstore."""
        llm = self.initialize_llm()
        prompt = self.create_prompt()
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),
            return_source_documents=True,
            verbose=True,
            combine_docs_chain_kwargs={"prompt": prompt}
        )

    def initialize_llm(self):
        if LLM_PROVIDER == "openai":
            return ChatOpenAI(temperature=0.3, model_name=LLM_MODEL, openai_api_key=self.openai_api_key)
        elif LLM_PROVIDER == "claude":
            return self.initialize_claude_llm()
        elif LLM_PROVIDER == "gemini":
            return self.initialize_gemini_llm()
        else:
            raise ValueError("Unsupported LLM_PROVIDER. Use 'openai', 'claude', or 'gemini'.")

    def initialize_claude_llm(self):
        self.validate_key(os.getenv("CLAUDE_API_KEY"), "CLAUDE_API_KEY")
        return ChatAnthropic(temperature=0.3, model=LLM_MODEL, anthropic_api_key=os.getenv("CLAUDE_API_KEY"))

    def initialize_gemini_llm(self):
        self.validate_key(os.getenv("GEMINI_API_KEY"), "GEMINI_API_KEY")
        return ChatGooglePalm(temperature=0.3, model_name=LLM_MODEL, google_api_key=os.getenv("GEMINI_API_KEY"))

    @staticmethod
    def create_prompt():
        return PromptTemplate(
            input_variables=["history", "context", "question"],
            template="""
            You're an assistant that answers questions strictly based on the provided documents.

            Conversation history:
            {history}

            Context from documents:
            {context}

            Question:
            {question}

            Answer:
            """
        )

    def answer_query(self, vectorstore, query):
        qa_chain = self.setup_qa_chain(vectorstore)
        # Format conversation history as a string
        history_text = "\n".join(
            [f"User: {q}\nAssistant: {a}" for q, a in conversation_history]
        )
        result = qa_chain({
            "question": query,
            "history": history_text,
            "chat_history": conversation_history
        })
        conversation_history.append((query, result["answer"]))
        return result["answer"]

# Create a global instance of RAGApp
rag_app = RAGApp()

@app.route('/process_files', methods=['POST'])
def process_files_endpoint():
    global vectorstore_global, chunks_global
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400
    files = request.files.getlist('files')
    try:
        chunks = rag_app.process_files(files)
        if not chunks:
            return jsonify({"error": "No valid files processed"}), 400
        chunks_global = chunks
        vectorstore_global = rag_app.initialize_vectorstore(chunks)
        return jsonify({"message": "Files processed successfully", "chunks_count": len(chunks)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_endpoint():
    global vectorstore_global, chunks_global
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' in request body"}), 400
    query = data["query"]
    if not vectorstore_global:
        if chunks_global:
            vectorstore_global = rag_app.initialize_vectorstore(chunks_global)
        else:
            return jsonify({"error": "No data processed. Please upload files first."}), 400
    try:
        answer = rag_app.answer_query(vectorstore_global, query)
        return jsonify({"query": query, "answer": answer}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def hello():
    return jsonify({"message": "Hello, Flask API!"})

if __name__ == '__main__':
    app.run(debug=True)
