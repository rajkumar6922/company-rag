"""Answer questions from a session's documents with retrieval + an OpenAI model."""

from dotenv import load_dotenv
from openai import OpenAI

from store import search

MODEL = "gpt-4.1-nano"

SYSTEM_PROMPT = (
    "You answer questions using only the context below. "
    "If the answer is not in the context, say you don't know. "
    "Keep answers brief and accurate. "
    "Mention the source file name of the facts you use."
)

EMPTY_MESSAGE = (
    "I don't have any documents to search yet. "
    "Please upload a document first, then ask your question."
)

# Read OPENAI_API_KEY from .env into the environment; OpenAI() picks it up
# from there. The key is never printed.
load_dotenv()
_client = OpenAI()


def _build_context(hits: list[dict]) -> str:
    # Label each chunk so the model can cite which file a fact came from.
    return "\n\n".join(
        f"[Source: {h['source']} | chunk {h['chunk_index']}]\n{h['text']}"
        for h in hits
    )


def answer_question(collection, question, history=None, k=5, debug=False):
    """Return (answer_text, sources, chunks) for a question about the session's documents.

    chunks is the list of retrieved chunks (dicts with text, source,
    chunk_index, distance) that was sent to the model as context.
    """
    # 1. Nothing uploaded yet: answer directly, no LLM call.
    if collection.count() == 0:
        return EMPTY_MESSAGE, [], []

    # 2. Retrieve the closest chunks.
    # NOTE: retrieval uses only the latest question, not the chat history, so
    # follow-ups like "what is her salary?" may fetch the wrong chunks because
    # "her" isn't resolved. We'll fix this later with query rewriting.
    hits = search(collection, question, k)

    # 3. System message = instructions + labeled context.
    context = _build_context(hits)
    system_message = f"{SYSTEM_PROMPT}\n\nContext:\n{context}"

    if debug:
        print("----- context sent to the model -----")
        print(system_message)
        print("-------------------------------------")

    # 4. System message, then earlier turns, then the new question.
    messages = [{"role": "system", "content": system_message}]
    messages += history or []
    messages.append({"role": "user", "content": question})

    response = _client.chat.completions.create(model=MODEL, messages=messages)
    answer = response.choices[0].message.content

    # 5. Sources = the distinct files whose chunks were given to the model,
    # in order of relevance.
    sources = list(dict.fromkeys(h["source"] for h in hits))
    return answer, sources, hits
