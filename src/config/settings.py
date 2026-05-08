import os
from dotenv import load_dotenv
import sys 


load_dotenv()  # loads from .env file if present 

#load root dir 
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, ROOT_DIR)

# ── Paths ──────────────────────────────────────────────
DOCUMENTS_FOLDER = os.path.join(ROOT_DIR, "data", "raw", "documents")
PROCESSED_FOLDER = os.path.join(ROOT_DIR, "data", "processed")
FAISS_INDEX_PATH = os.path.join(ROOT_DIR, "vector_db", "faiss_index", "index.faiss")
FAISS_META_PATH  = os.path.join(ROOT_DIR, "vector_db", "faiss_index", "metadata.pkl")
CHROMA_PATH      = os.path.join(ROOT_DIR, "vector_db", "chroma_db")

# ── Chunking ───────────────────────────────────────────
CHUNK_SIZE         = 200
CHUNK_OVERLAP      = 80

# ── Embeddings ─────────────────────────────────────────
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
EMBEDDING_DIM      = 384

# ── Retrieval ──────────────────────────────────────────
TOP_K              = 5
MIN_SCORE     = 0.15
VECTOR_STORE       = "faiss"   # "faiss" or "chroma"

# ── LLM ────────────────────────────────────────────────
LLM_PROVIDER       = os.getenv("LLM_PROVIDER", "groq")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL         = "llama-3.1-8b-instant"  # free, fast, very capable
OLLAMA_MODEL       = "mistral"          # fallback if no internet
OLLAMA_BASE_URL    = "http://localhost:11434"
MAX_TOKENS         = 512
TEMPERATURE        = 0.2

# ── ML ─────────────────────────────────────────────────
ML_MODEL_PATH = os.path.join(ROOT_DIR, "models", "ml_models",
                              "doc_classifier.pkl")
TRAIN_CSV     = os.path.join(ROOT_DIR, "data", "annotations",
                              "classification_labels", "train_data.csv")
TEST_CSV      = os.path.join(ROOT_DIR, "data", "annotations",
                              "classification_labels", "test_data.csv")