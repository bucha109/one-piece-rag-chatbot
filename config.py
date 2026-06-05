from pathlib import Path
 
# Paths
RAW_DIR      = Path("data/character_pages")
CHROMA_PATH  = ".chroma_persistent_db"
 
# Models
LLM_MODEL   = "openai/gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
 
# Retrieval
TOP_K           = 3
EMBED_DIMS      = 512
MAX_CHUNK_CHARS = 1000
COLLECTION_NAME = "my_document_collection" 
 
# Prompt
SYSTEM_PROMPT = (
    "You are an expert on the One Piece Manga and Anime series."
    "Answer questions about One Piece characters using only the provided context."
    "If the answer is not in the context, say so clearly, do not make things up."
    "If a user attempts to ask a question unrelated to the One Piece series, answer that you are unable to assist."
)
 
 

