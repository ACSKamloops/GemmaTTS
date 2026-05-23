"""Sentence-aware text chunking for TTS synthesis.

Splits long text into chunks that respect sentence boundaries and stay
within a configurable character limit.  This prevents TTS engines from
receiving inputs that are too long (which degrades quality or causes
timeouts), while avoiding mid-sentence splits that produce unnatural
prosody.

Typical usage:
    from app.text import chunk_text
    chunks = chunk_text("Hello world. This is a test.", max_chars=250)
"""

import logging
import re
from typing import List

logger = logging.getLogger("tts-chunker")

# Default maximum characters per chunk — tuned for Kokoro, which works
# best with inputs around 200-250 characters.
DEFAULT_MAX_CHARS = 250

# Primary split: sentence-ending punctuation followed by whitespace, or CJK sentence-ending punctuation.
_SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+|(?<=[。！？；])")

# Secondary split: commas followed by whitespace, or CJK commas.
_COMMA_RE = re.compile(r"(?<=,)\s+|(?<=[，、])")


def _split_at_pattern(text: str, pattern: re.Pattern) -> List[str]:
    """Split *text* using a regex pattern, keeping the delimiter attached
    to the preceding fragment (look-behind in the pattern ensures this)."""
    parts = pattern.split(text)
    return [p for p in parts if p]


def _force_split(text: str, max_chars: int) -> List[str]:
    """Last-resort word-level split for fragments that exceed *max_chars*
    even after comma splitting.

    Walks through words and accumulates them into a buffer.  When adding
    the next word would exceed the limit the current buffer is flushed.
    A single word longer than *max_chars* is emitted as-is (we never
    break within a word), unless max_chars is 1, in which case we must
    split.
    """
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    for word in words:
        if len(word) > max_chars:
            if max_chars == 1:
                if buf:
                    chunks.append(" ".join(buf))
                    buf = []
                    buf_len = 0
                for char in word:
                    chunks.append(char)
                continue
            # Otherwise, emit word as-is (will be added to chunks or buffer)

        # +1 accounts for the space that joins words
        added_len = len(word) + (1 if buf else 0)
        if buf and buf_len + added_len > max_chars:
            chunks.append(" ".join(buf))
            buf = [word]
            buf_len = len(word)
        else:
            buf.append(word)
            buf_len += added_len

    if buf:
        chunks.append(" ".join(buf))

    return chunks


def _split_long_fragment(fragment: str, max_chars: int) -> List[str]:
    """Break a single sentence/fragment that exceeds *max_chars*.

    Strategy (ordered):
        1. Try splitting at commas.
        2. If any sub-fragment is still too long, force-split at word
           boundaries.
    """
    if len(fragment) <= max_chars:
        return [fragment]

    # Try comma boundaries first
    sub_parts = _split_at_pattern(fragment, _COMMA_RE)
    result: List[str] = []
    for part in sub_parts:
        if len(part) <= max_chars:
            result.append(part)
        else:
            result.extend(_force_split(part, max_chars))
    return result


def _merge_fragments(fragments: List[str], max_chars: int) -> List[str]:
    """Greedily merge small fragments into chunks without exceeding
    *max_chars*.  A single space is used to join fragments within a chunk.
    """
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    for frag in fragments:
        added_len = len(frag) + (1 if buf else 0)
        if buf and buf_len + added_len > max_chars:
            chunks.append(" ".join(buf))
            buf = [frag]
            buf_len = len(frag)
        else:
            buf.append(frag)
            buf_len += added_len

    if buf:
        chunks.append(" ".join(buf))

    return chunks


def chunk_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> List[str]:
    """Split *text* into TTS-friendly chunks.

    Parameters
    ----------
    text : str
        The input text to chunk.  Leading/trailing whitespace is stripped
        and internal whitespace is normalised.
    max_chars : int, optional
        Maximum character count per chunk (default 250).

    Returns
    -------
    list[str]
        Ordered list of non-empty text chunks.  Empty or whitespace-only
        input yields an empty list.

    Examples
    --------
    >>> chunk_text("Hi.")
    ['Hi.']
    >>> chunk_text("")
    []
    """
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")

    # Normalise whitespace
    text = " ".join(text.split())
    if not text:
        return []

    # Fast path: already under the limit
    if len(text) <= max_chars:
        return [text]

    # 1. Split at sentence boundaries
    sentences = _split_at_pattern(text, _SENTENCE_RE)

    # 2. Break any oversized sentences further
    fragments: List[str] = []
    for sent in sentences:
        if len(sent) <= max_chars:
            fragments.append(sent)
        else:
            fragments.extend(_split_long_fragment(sent, max_chars))

    # 3. Merge small adjacent fragments to fill chunks efficiently
    chunks = _merge_fragments(fragments, max_chars)

    logger.debug(
        "Chunked %d-char input into %d chunks (max_chars=%d)",
        len(text),
        len(chunks),
        max_chars,
    )
    return chunks
