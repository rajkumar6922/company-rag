"""Load, chunk and store every file in samples/, then run a few searches."""

from pathlib import Path

from chunker import chunk_text
from loader import load_text
from store import add_document, create_collection, search

SAMPLES_DIR = Path(__file__).parent / "samples"

QUESTIONS = [
    "How many days of paid leave do employees get?",
    "Where are the warehouses and how many pallets can they hold?",
    "What does the warranty cover and how long does a claim take?",
]

collection = create_collection("test-session")

# Load and chunk each sample file, then store its chunks.
for file in sorted(SAMPLES_DIR.iterdir()):
    if not file.is_file():
        continue
    try:
        text = load_text(file)
    except ValueError as e:
        print(f"Skipped {file.name}: {e}")
        continue
    chunks = chunk_text(text)
    add_document(collection, file.name, chunks)
    print(f"Stored {len(chunks)} chunks from {file.name}")

print(f"Collection holds {collection.count()} chunks\n")

# Ask each question and show the top 3 matches.
for question in QUESTIONS:
    print("=" * 60)
    print(f"Q: {question}")
    for rank, hit in enumerate(search(collection, question, k=3), start=1):
        snippet = hit["text"][:150].replace("\n", " ")
        print(
            f"  {rank}. {hit['source']} (chunk {hit['chunk_index']}) "
            f"distance={hit['distance']:.3f}"
        )
        print(f"     {snippet}")
