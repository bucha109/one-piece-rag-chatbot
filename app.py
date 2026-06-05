### App.py will be the orchestrator of the entire end to end RAG system

import gradio as gr 
import chromadb
from openai import OpenAI
from pathlib import Path

from fetch_pages import fetch_pages
from embeddings import collection, retrieve, ai_client, index_chunks, visualize_embeddings_3d
from chunking import load_and_chunk
from prompt import build_prompt, generate_answer

from config import TOP_K, RAW_DIR


def build_ui(collection: chromadb.Collection, llm_client: OpenAI) -> gr.ChatInterface:
    """
    Construct and return the Gradio ChatInterface.
 
    Chat function flow per turn:
        user query → retrieve chunks → build prompt → generate → display
    """
    fig = visualize_embeddings_3d(collection)

    def chat(user_message: str, history: list[dict]) -> str: #  runs every time the user sends a message in the Gradio UI.
        chunks  = retrieve(collection, user_message, TOP_K)
        prompt  = build_prompt(user_message, chunks)
        answer  = generate_answer(llm_client, prompt, history)
        sources = '\n'.join(
            f'{c["source"]} \n {c["section"]} \n {c["subsection"]}\n {c["text"][:250]}'
            for c in chunks
            )
        return f"{answer}\n\n*Sources: {sources}*"

    with gr.Blocks(theme=gr.themes.Ocean()) as demo:
        gr.ChatInterface(
            fn=chat, # call the chat function whenever the user submits a message
            title="One Piece Character Chatbot",
            chatbot=gr.Chatbot(height=460), 
        )
        gr.Plot(value=fig)



    return demo

def main():
    # One time setup
    if not any(RAW_DIR.glob("*.txt")):
        print("No character data present, fetching")
        fetch_pages()
    chunks = load_and_chunk(RAW_DIR)
    index_chunks(collection, chunks)

    demo = build_ui(collection, ai_client)
    demo.launch()


if __name__ == "__main__":
    main()