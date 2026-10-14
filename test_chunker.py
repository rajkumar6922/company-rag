"""Chunk every file in samples/ and print chunk stats plus the first two chunks."""

from pathlib import Path

from chunker import chunk_text
from loader import load_text

SAMPLES_DIR = Path(__file__).parent / "samples"

for file in sorted(SAMPLES_DIR.iterdir()):
    if not file.is_file():
        continue
    print("=" * 60)
    print(f"File: {file.name}")
    try:
        text = load_text(file)
    except ValueError as e:
        print(f"Skipped: {e}")
        continue

    chunks = chunk_text(text)
    lengths = [len(c) for c in chunks]
    print(f"Chunks: {len(chunks)}")
    if not chunks:
        continue
    print(
        f"Length  min: {min(lengths)}  avg: {sum(lengths) / len(lengths):.0f}  "
        f"max: {max(lengths)}"
    )

    # Show the first two chunks so the overlap is visible.
    for i in range(min(2, len(chunks))):
        print(f"--- chunk {i} ({lengths[i]} chars) ---")
        print(chunks[i])
