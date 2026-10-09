# Company RAG

A Retrieval-Augmented Generation (RAG) application that answers questions using a
company's own documents. Documents in `data/knowledge-base/` are split into chunks,
embedded into a vector database, and the most relevant chunks are passed to an LLM
as context so answers are grounded in real company knowledge.

## Project structure

```
company-rag/
├── src/rag_app/          # Application source code
├── data/knowledge-base/  # Documents the app answers questions about
├── tests/                # Automated tests
├── requirements.txt      # Python dependencies
├── .env.example          # Template for environment variables
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
copy .env.example .env        # then add your API key to .env
```

## Running tests

```bash
pytest
```
