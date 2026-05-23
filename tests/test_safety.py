import pytest
from app.safety.text_sanitizer import sanitize_text
from app.safety.output_validator import validate_llm_json

def test_sanitize_urls():
    assert sanitize_text("Hello http://example.com world") == "Hello world"
    assert sanitize_text("Check https://google.com/abc?q=123 out") == "Check out"
    assert sanitize_text("Link: [Google](https://google.com)") == "Link: Google"

def test_sanitize_html():
    assert sanitize_text("Hello <b>world</b>") == "Hello world"
    assert sanitize_text("<script>alert(1)</script> Hello") == "alert(1) Hello"

def test_sanitize_markdown():
    assert sanitize_text("**Bold** and *italic* text") == "Bold and italic text"
    assert sanitize_text("Code `print('hello')` here") == "Code print('hello') here"
    assert sanitize_text("# Heading 1\n## Heading 2") == "Heading 1 Heading 2"

def test_sanitize_traversal_attempts():
    assert sanitize_text("Hello ../../../etc/passwd world") == "Hello etc/passwd world"
    assert sanitize_text("Hello ..\\..\\..\\windows world") == "Hello windows world"

def test_sanitize_limits():
    # Enforce word limits
    long_text = " ".join(["word"] * 200)
    sanitized = sanitize_text(long_text)
    assert len(sanitized.split()) == 150 # max_text_words config limit

def test_validate_llm_json_valid():
    raw = '{"text": "Keep your voice down."}'
    data = validate_llm_json(raw)
    assert data is not None
    assert data["text"] == "Keep your voice down."

def test_validate_llm_json_markdown():
    raw = """
Some conversational text before the block.
```json
{
  "text": "Keep your voice down."
}
```
Some text after.
"""
    data = validate_llm_json(raw)
    assert data is not None
    assert data["text"] == "Keep your voice down."

def test_validate_llm_json_malformed():
    raw = '{"text": "Keep your voice down."' # missing brace
    assert validate_llm_json(raw) is None

def test_validate_llm_json_invalid_schema():
    raw = '{"content": "Keep your voice down."}' # wrong key
    assert validate_llm_json(raw) is None
