from typing import Optional, List, Dict, Any

def build_dialogue_prompt(
    user_text: str,
    speaker_name: str,
    speaker_style: Optional[str] = None,
    location: Optional[str] = None,
    facts: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Constructs a schema-constrained prompt instructing the LLM
    to respond strictly in JSON format matching the DialogueResponseSchema.
    """
    system_instruction = (
        "You are an assistant participating in a dialogue. "
        "You must respond STRICTLY in JSON format with a single key 'text' containing your spoken response. "
        "Do not include any conversational explanation, markdown formatting, or text outside the JSON object.\n"
        "JSON Schema:\n"
        "{\n"
        "  \"text\": \"string (your spoken response)\"\n"
        "}\n"
    )
    
    context_parts = []
    if speaker_name:
        context_parts.append(f"Speaker Name: {speaker_name}")
    if speaker_style:
        context_parts.append(f"Speaker Style: {speaker_style}")
    if location:
        context_parts.append(f"Location: {location}")
        
    if facts:
        revealed = [f.get("fact", "") for f in facts if f.get("can_reveal", False)]
        if revealed:
            context_parts.append("Background Facts:\n" + "\n".join(f"- {fact}" for fact in revealed))
            
    context_str = "\n".join(context_parts)
    
    prompt = (
        f"{system_instruction}\n"
        f"Context:\n{context_str}\n\n"
        f"User: {user_text}\n"
        f"Response (JSON):"
    )
    return prompt
