"""Store every file in samples/, then ask questions through answer_question."""

from pathlib import Path

from answer import answer_question
from chunker import chunk_text
from loader import load_text
from store import add_document, create_collection

SAMPLES_DIR = Path(__file__).parent / "samples"

QUESTIONS = [
    "How many days of paid leave do Northwind Lumen employees get?",
    "Which Bluefin warehouse has the largest capacity?",
    "How long does Northwind Lumen take to handle a warranty claim?",
    # Not in any document: the model should say it doesn't know.
    "What is the CEO of Bluefin Logistics called?",
]

collection = create_collection("test-session")

# The empty-collection case should answer without calling the model.
print("Empty collection:", answer_question(collection, "Anything?")[0], "\n")

# Load, chunk and store each sample file.
for file in sorted(SAMPLES_DIR.iterdir()):
    if not file.is_file():
        continue
    try:
        add_document(collection, file.name, chunk_text(load_text(file)))
    except ValueError as e:
        print(f"Skipped {file.name}: {e}")

for question in QUESTIONS:
    answer, sources, chunks = answer_question(collection, question)
    print("=" * 60)
    print(f"Q: {question}")
    print(f"A: {answer}")
    print(f"Sources: {', '.join(sources)}")
    print(f"Chunks:  {', '.join(f'{c['source']}#{c['chunk_index']}' for c in chunks)}")
