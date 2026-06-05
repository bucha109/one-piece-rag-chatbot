from openai import OpenAI
from config import LLM_MODEL, SYSTEM_PROMPT


def build_prompt(query: str, context_chunks: list[dict]) -> str:
    """
    Assemble the prompt sent to the Responses API.
    Structure:
        [Retrieved context chunks]
        [User question]
    The system instruction is passed separately via `instructions` in
    generate_answer(), keeping system and user content clearly separated.
    """
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']} \n {c['section']} \n {c['subsection']} \n {c['text']}" for c in context_chunks
    ) 
    return (
        f"### Retrieved Context:\n{context}\n\n"
        f"### Question:\n{query}"
    )


def generate_answer(client: OpenAI, prompt: str, history: list[dict]) -> str:
    """
    Send the prompt to the OpenAI Responses API and return the reply.
 
    • instructions  — system-level persona and guardrails (separate from input)
    • input         — retrieved context + user question
    • output_text   — clean string response; no unpacking required
    """
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content" : SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip() # remove leading/trailing whitespace from the response. Removers spaces, newlines, tabs, etc. 
