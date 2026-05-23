"""Unit tests for app.text.chunker — sentence-aware TTS text chunking."""

import pytest

from app.text.chunker import chunk_text, DEFAULT_MAX_CHARS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_under_limit(chunks: list[str], limit: int) -> bool:
    """Assert every chunk is at most *limit* characters."""
    return all(len(c) <= limit for c in chunks)


def _reconstructs(chunks: list[str], original: str) -> bool:
    """Verify that joining the chunks reproduces the normalised original."""
    return " ".join(chunks) == " ".join(original.split())


# ---------------------------------------------------------------------------
# Basic / happy-path
# ---------------------------------------------------------------------------

class TestBasicChunking:
    def test_short_text_single_chunk(self):
        text = "Hello, world."
        result = chunk_text(text)
        assert result == ["Hello, world."]

    def test_exact_limit_returns_single_chunk(self):
        text = "a" * 250
        result = chunk_text(text, max_chars=250)
        assert result == [text]

    def test_single_word(self):
        result = chunk_text("Hello")
        assert result == ["Hello"]

    def test_default_max_chars_is_250(self):
        assert DEFAULT_MAX_CHARS == 250


# ---------------------------------------------------------------------------
# Empty / whitespace input
# ---------------------------------------------------------------------------

class TestEmptyInput:
    def test_empty_string(self):
        assert chunk_text("") == []

    def test_whitespace_only(self):
        assert chunk_text("   \t\n  ") == []

    def test_none_like_empty(self):
        # Passing a genuinely empty string is the canonical "no input" case.
        assert chunk_text("") == []


# ---------------------------------------------------------------------------
# Sentence-boundary splitting
# ---------------------------------------------------------------------------

class TestSentenceSplitting:
    def test_splits_at_period(self):
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunk_text(text, max_chars=40)
        assert len(chunks) >= 2
        assert _all_under_limit(chunks, 40)
        assert _reconstructs(chunks, text)

    def test_splits_at_exclamation(self):
        text = "Wow! That was great! Let me tell you more!"
        chunks = chunk_text(text, max_chars=25)
        assert len(chunks) >= 2
        assert _all_under_limit(chunks, 25)

    def test_splits_at_question_mark(self):
        text = "How are you? What is this? Where do we go?"
        chunks = chunk_text(text, max_chars=25)
        assert len(chunks) >= 2
        assert _all_under_limit(chunks, 25)

    def test_splits_at_semicolon(self):
        text = "Part one; part two; part three."
        chunks = chunk_text(text, max_chars=20)
        assert len(chunks) >= 2
        assert _all_under_limit(chunks, 20)

    def test_mixed_punctuation(self):
        text = "Hello! How are you? Fine, thanks. Really; I mean it."
        chunks = chunk_text(text, max_chars=30)
        assert len(chunks) >= 2
        assert _all_under_limit(chunks, 30)
        assert _reconstructs(chunks, text)


# ---------------------------------------------------------------------------
# Sentence grouping / merging
# ---------------------------------------------------------------------------

class TestMerging:
    def test_groups_short_sentences(self):
        """Short sentences that individually fit should be merged."""
        text = "Hi. Ok. Go."
        result = chunk_text(text, max_chars=250)
        # All three fit comfortably in one chunk
        assert result == ["Hi. Ok. Go."]

    def test_groups_until_limit(self):
        text = "One. Two. Three. Four. Five. Six. Seven. Eight."
        chunks = chunk_text(text, max_chars=30)
        assert _all_under_limit(chunks, 30)
        assert _reconstructs(chunks, text)


# ---------------------------------------------------------------------------
# Long single sentences (fallback splitting)
# ---------------------------------------------------------------------------

class TestLongSentenceFallback:
    def test_splits_at_commas(self):
        text = "This is a long sentence, with several clauses, that should be split, at comma boundaries."
        chunks = chunk_text(text, max_chars=50)
        assert len(chunks) >= 2
        assert _all_under_limit(chunks, 50)
        assert _reconstructs(chunks, text)

    def test_force_splits_at_spaces(self):
        # No commas, no sentence-ending punctuation — pure word split
        text = "word " * 60  # 300 chars
        chunks = chunk_text(text.strip(), max_chars=50)
        assert len(chunks) >= 2
        assert _all_under_limit(chunks, 50)
        assert _reconstructs(chunks, text.strip())

    def test_single_very_long_word_emitted_as_is(self):
        """A word longer than max_chars can't be split — returned intact."""
        long_word = "a" * 300
        chunks = chunk_text(long_word, max_chars=50)
        assert chunks == [long_word]

    def test_long_word_mixed_with_short_words(self):
        long_word = "b" * 100
        text = f"Short. {long_word} end."
        chunks = chunk_text(text, max_chars=50)
        # The long word must appear verbatim in exactly one chunk
        assert any(long_word in c for c in chunks)
        assert _reconstructs(chunks, text)


# ---------------------------------------------------------------------------
# Max chars enforcement
# ---------------------------------------------------------------------------

class TestMaxCharsEnforcement:
    def test_all_chunks_within_limit(self):
        text = (
            "The quick brown fox jumped over the lazy dog. "
            "Pack my box with five dozen liquor jugs. "
            "How vexingly quick daft zebras jump! "
            "The five boxing wizards jump quickly."
        )
        for limit in [30, 50, 80, 120, 200]:
            chunks = chunk_text(text, max_chars=limit)
            assert _all_under_limit(chunks, limit), (
                f"Failed for max_chars={limit}: {[len(c) for c in chunks]}"
            )
            assert _reconstructs(chunks, text)

    def test_max_chars_one(self):
        """With max_chars=1 every single character becomes its own chunk
        (except spaces which are joining glue)."""
        result = chunk_text("ab", max_chars=1)
        assert result == ["a", "b"]

    def test_invalid_max_chars_raises(self):
        with pytest.raises(ValueError, match="max_chars must be >= 1"):
            chunk_text("hello", max_chars=0)


# ---------------------------------------------------------------------------
# Unicode / international text
# ---------------------------------------------------------------------------

class TestUnicode:
    def test_japanese_text(self):
        text = "こんにちは世界。これはテストです。"
        chunks = chunk_text(text, max_chars=15)
        assert len(chunks) >= 1
        assert _all_under_limit(chunks, 15)

    def test_emoji_text(self):
        text = "Hello! 😀 How are you? 🎉 Great!"
        chunks = chunk_text(text, max_chars=20)
        assert len(chunks) >= 1
        assert _all_under_limit(chunks, 20)

    def test_mixed_scripts(self):
        text = "English text. Текст на русском. 中文文本。"
        chunks = chunk_text(text, max_chars=30)
        assert len(chunks) >= 1
        assert _all_under_limit(chunks, 30)


# ---------------------------------------------------------------------------
# Whitespace normalisation
# ---------------------------------------------------------------------------

class TestWhitespaceNormalisation:
    def test_multiple_spaces_collapsed(self):
        text = "Hello   world.   How   are   you?"
        result = chunk_text(text, max_chars=250)
        assert result == ["Hello world. How are you?"]

    def test_newlines_collapsed(self):
        text = "Line one.\nLine two.\nLine three."
        result = chunk_text(text, max_chars=250)
        assert result == ["Line one. Line two. Line three."]

    def test_tabs_collapsed(self):
        text = "Tab\there.\tAnother\tline."
        result = chunk_text(text, max_chars=250)
        assert result == ["Tab here. Another line."]


# ---------------------------------------------------------------------------
# Preservation / reconstruction
# ---------------------------------------------------------------------------

class TestReconstruction:
    def test_round_trip_preserves_all_content(self):
        text = (
            "This is sentence one. This is sentence two! "
            "Is this sentence three? Yes; it is. "
            "And here comes sentence five."
        )
        for limit in [20, 50, 100, 250]:
            chunks = chunk_text(text, max_chars=limit)
            assert _reconstructs(chunks, text), (
                f"Content lost at max_chars={limit}"
            )


# ---------------------------------------------------------------------------
# Import via package init
# ---------------------------------------------------------------------------

class TestPackageExport:
    def test_import_from_package(self):
        from app.text import chunk_text as ct
        assert callable(ct)
        assert ct("Hello.") == ["Hello."]
