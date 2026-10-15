---
title: DocLens
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# DocLens

DocLens lets you upload your own documents and ask questions about them. Every answer is grounded in the uploaded text and cites the files it came from.

## How it works

1. **Upload**: PDF, DOCX, TXT or MD (up to 3 files, 5 MB each).
2. **Chunk**: text is split into ~500-character chunks with 100 characters of overlap.
3. **Embed**: each chunk is embedded locally with `all-MiniLM-L6-v2` (384 dimensions).
4. **Store**: embeddings go into an in-memory Chroma collection, one per browser session.
5. **Retrieve**: the top 5 chunks by cosine similarity are fetched for each question.
6. **Answer**: `gpt-4.1-nano` answers using only the retrieved context and lists its sources.

Uploaded files live only in memory for the current session. Retrieved text is sent to OpenAI to write answers.

## Tech stack

- Python 3.12
- Gradio (web UI)
- ChromaDB (in-memory vector database)
- all-MiniLM-L6-v2 (embeddings, via Chroma's ONNX default)
- OpenAI gpt-4.1-nano (answer generation)
- pypdf / python-docx (document parsing)
- uv (package management)
- Hugging Face Spaces (hosting)

## Run locally

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/rajkumar6922/company-rag.git
cd company-rag
uv sync
echo "OPENAI_API_KEY=your-key-here" > .env
uv run python app.py
```

Then open the URL Gradio prints (usually http://127.0.0.1:7860).

On Hugging Face Spaces, set `OPENAI_API_KEY` as a secret in the Space settings instead of using a `.env` file.

## Evaluation results

_To be added._
