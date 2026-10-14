"""Print a quick summary of every file in samples/ using load_text."""

from pathlib import Path

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
        # Unsupported or scanned files are reported, not fatal.
        print(f"Skipped: {e}")
        continue
    print(f"Characters: {len(text)}")
    print("First 300 characters:")
    print(text[:300])
