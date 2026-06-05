# One Piece Character Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about One Piece characters. Built with ChromaDB, OpenAI embeddings, and a Gradio UI, includes an interactive 3D visualization of the embedding space.

## How It Works

1. **Fetch** — `fetch_pages.py` scrapes Wikipedia articles for a configurable set of One Piece characters and saves them as structured `.txt` files.
2. **Chunk** — `chunking.py` splits each article into semantically meaningful chunks, preserving section and subsection metadata.
3. **Embed & Index** — `embeddings.py` generates embeddings and stores them in a persistent ChromaDB collection.
4. **Retrieve & Answer** — `prompt.py` retrieves the top-k most relevant chunks for a query and passes them as context to `gpt-4o-mini`.
5. **UI** — `app.py` orchestrates everything and launches a Gradio chat interface alongside an interactive UMAP 3D scatter plot of the embedding space.

## Characters Covered

The Straw Hat crew (Luffy, Zoro, Nami, Usopp, Sanji, Chopper, Robin, Franky, Brook) plus the Four Emperors. Additional characters can be added by editing `characters.yaml`.

## Setup

**Prerequisites:** Python 3.11+

```bash
# 1. Clone the repo
git clone https://github.com/your-username/one-piece-rag-chatbot.git
cd one-piece-rag-chatbot

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

**.env format:**
```
OPENROUTER_API_KEY=your_key_here
```

> An [OpenRouter](https://openrouter.ai) account is required. The project uses `openai/gpt-4o-mini` for generation and `text-embedding-3-small` for embeddings, both routed through OpenRouter.

## Running

```bash
python app.py
```

On first run, the app will fetch Wikipedia pages and build the ChromaDB index automatically. Subsequent runs reuse the cached index. The Gradio interface will open in your browser.

## Project Structure

```
├── app.py              # Orchestrator + Gradio UI
├── fetch_pages.py      # Wikipedia scraper
├── chunking.py         # Text chunking logic
├── embeddings.py       # Embedding, indexing, retrieval, and 3D visualization
├── prompt.py           # Prompt construction and LLM call
├── characters.yaml     # List of characters to scrape
├── requirements.txt
└── .env.example
```

## Tech Stack

- **LLM:** GPT-4o-mini (via OpenRouter)
- **Embeddings:** text-embedding-3-small (512 dimensions)
- **Vector Store:** ChromaDB (persistent)
- **Dimensionality Reduction:** UMAP → 3D
- **Clustering:** K-Means
- **UI:** Gradio
