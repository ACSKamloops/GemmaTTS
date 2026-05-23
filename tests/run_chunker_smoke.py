"""Quick smoke test for the chunker, run directly with python."""
import sys
sys.path.insert(0, '.')
from app.text.chunker import chunk_text

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name} - {detail}")
        failed += 1

# Test 1
r = chunk_text("Hello, world.")
check("short text single chunk", r == ["Hello, world."], str(r))

# Test 2
r = chunk_text("")
check("empty string", r == [], str(r))

# Test 3
r = chunk_text("   \t\n  ")
check("whitespace only", r == [], str(r))

# Test 4
r = chunk_text("First sentence. Second sentence. Third sentence.", max_chars=40)
check("sentence splitting count", len(r) >= 2, str(r))
check("sentence splitting limit", all(len(c) <= 40 for c in r), str([len(c) for c in r]))
check("sentence splitting reconstruct", " ".join(r) == "First sentence. Second sentence. Third sentence.", " ".join(r))

# Test 5
text = "This is a long sentence, with several clauses, that should be split, at comma boundaries."
r = chunk_text(text, max_chars=50)
check("comma split count", len(r) >= 2, str(r))
check("comma split limit", all(len(c) <= 50 for c in r), str([len(c) for c in r]))

# Test 6
text = " ".join(["word"] * 60)
r = chunk_text(text, max_chars=50)
check("word split count", len(r) >= 2, str(r))
check("word split limit", all(len(c) <= 50 for c in r), str([len(c) for c in r]))

# Test 7
long_word = "a" * 300
r = chunk_text(long_word, max_chars=50)
check("very long word passthrough", r == [long_word], str(r))

# Test 8
r = chunk_text("Hello! How are you? Fine, thanks. Really; I mean it.", max_chars=30)
check("mixed punctuation", len(r) >= 2 and all(len(c) <= 30 for c in r), str(r))

# Test 9
r = chunk_text("Hello! \U0001f600 How are you? \U0001f389 Great!", max_chars=20)
check("unicode emoji", len(r) >= 1 and all(len(c) <= 20 for c in r), str(r))

# Test 10
text = "This is sentence one. This is sentence two! Is this sentence three? Yes; it is."
r = chunk_text(text, max_chars=50)
check("round-trip reconstruction", " ".join(r) == " ".join(text.split()), " ".join(r))

# Test 11
try:
    chunk_text("hello", max_chars=0)
    check("ValueError on max_chars=0", False, "no exception")
except ValueError:
    check("ValueError on max_chars=0", True)

# Test 12
r = chunk_text("Hello   world.   How   are   you?")
check("whitespace normalisation", r == ["Hello world. How are you?"], str(r))

# Test 13
from app.text import chunk_text as ct
check("package export works", callable(ct) and ct("Hi.") == ["Hi."])

print(f"\n{passed} passed, {failed} failed out of {passed + failed} tests")
sys.exit(1 if failed else 0)
