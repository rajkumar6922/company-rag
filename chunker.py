"""Split text into overlapping chunks that end at natural breaks."""

import re

# A sentence ends with . ! or ?, optionally followed by a closing quote or
# bracket, and then whitespace.
_SENTENCE_END = re.compile(r"[.!?][\"')\]]?\s")


def _find_cut(text: str, lo: int, hi: int) -> int:
    """Return the best place to end a chunk inside text[lo:hi].

    Preference: blank line, then sentence end, then space, then hi itself
    (a hard cut in the middle of a word).
    """
    window = text[lo:hi]

    # 1. Last blank line in the window.
    pos = window.rfind("\n\n")
    if pos != -1:
        return lo + pos

    # 2. Last sentence end in the window (cut right after the punctuation).
    last = None
    for last in _SENTENCE_END.finditer(window):
        pass
    if last is not None:
        return lo + last.end()

    # 3. Last whitespace in the window.
    pos = max(window.rfind(" "), window.rfind("\n"))
    if pos != -1:
        return lo + pos

    # 4. No natural break: cut mid-word.
    return hi


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Cut text into chunks of about chunk_size characters that overlap."""
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
        )

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)

        if end == n:
            # The rest of the text fits in one chunk.
            cut = n
        else:
            # Only look for a break past start + overlap, so the next chunk
            # (which starts overlap chars before the cut) is always further on.
            cut = _find_cut(text, start + overlap + 1, end)

        chunk = text[start:cut].strip()
        if chunk:
            chunks.append(chunk)

        if cut >= n:
            break

        # Next chunk starts overlap characters before this cut. Skip ahead to
        # the next word boundary so it doesn't begin halfway through a word.
        next_start = cut - overlap
        space = text.find(" ", next_start, cut)
        if space != -1:
            next_start = space + 1

        # Safety net: always move forward.
        start = max(next_start, start + 1)

    return chunks
