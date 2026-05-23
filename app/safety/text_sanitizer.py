import re
from app.config import settings

# Regex to match markdown links [text](url) and extract text
MARKDOWN_LINK_REGEX = re.compile(r'\[(.*?)\]\((?:https?|ftp|file)://[^\s/$.?#].[^\s]*\)')

# Regex to detect raw URLs
RAW_URL_REGEX = re.compile(r'(?:https?|ftp|file)://[^\s/$.?#].[^\s]*')

# Regex to detect HTML tags
HTML_REGEX = re.compile(r'<[^>]*>')

# Markdown formatting characters to strip (including backslash)
MARKDOWN_CHARS = re.compile(r'[*_`#~\[\]\\]')

# Path traversal regex to detect sequences like ../ or ..\
TRAVERSAL_REGEX = re.compile(r'\.\.+[/\\]')

def sanitize_text(text: str) -> str:
    """
    Sanitizes text by removing URLs (preserving markdown link text), HTML,
    and path traversals, then stripping markdown formatting and enforcing limits.
    """
    if not text:
        return ""
        
    # Preserve markdown link text but strip the URL
    text = MARKDOWN_LINK_REGEX.sub(r"\1", text)
    
    # Strip any remaining raw URLs
    text = RAW_URL_REGEX.sub("", text)
    
    # Remove HTML tags
    text = HTML_REGEX.sub("", text)
    
    # Remove traversal sequences (done before stripping markdown backslashes)
    text = TRAVERSAL_REGEX.sub("", text)
    
    # Strip markdown format characters
    text = MARKDOWN_CHARS.sub("", text)
    
    # Normalise whitespace
    text = " ".join(text.split())
    
    # Enforce character limit
    if len(text) > settings.max_text_chars:
        text = text[:settings.max_text_chars]
        
    # Enforce word limit
    words = text.split()
    if len(words) > settings.max_text_words:
        text = " ".join(words[:settings.max_text_words])
        
    return text

