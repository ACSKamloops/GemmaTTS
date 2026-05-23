import json
import re
from typing import Dict, Any, Optional
from pydantic import BaseModel, ValidationError

class DialogueResponseSchema(BaseModel):
    text: str

def validate_llm_json(raw_output: str) -> Optional[Dict[str, Any]]:
    """
    Extracts and parses JSON from the LLM output, enforcing a schema
    that contains at least a valid 'text' field.
    """
    if not raw_output:
        return None
        
    cleaned = raw_output.strip()
    
    # Check for markdown json block
    markdown_json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
    if markdown_json_match:
        cleaned = markdown_json_match.group(1).strip()
    else:
        # Find first '{' and last '}' to strip surrounding conversational chatter
        first_brace = cleaned.find('{')
        last_brace = cleaned.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            cleaned = cleaned[first_brace:last_brace + 1].strip()
            
    try:
        data = json.loads(cleaned)
        DialogueResponseSchema(**data)
        return data
    except (json.JSONDecodeError, ValidationError):
        return None
