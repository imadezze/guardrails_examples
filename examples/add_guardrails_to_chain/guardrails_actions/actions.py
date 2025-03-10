
from nemoguardrails.actions import action

@action
async def check_input_toxicity(input_text: str = ""):
    """
    Check if the input text is toxic.
    """
    harmful_keywords = [
        "hack", "steal", "illegal", "bomb", "kill", "hurt", "attack",
        "exploit", "cheat", "fraud", "weapon", "violent", "abuse"
    ]
    
    input_lower = input_text.lower()
    is_toxic = any(keyword in input_lower for keyword in harmful_keywords)
    
    return {"is_toxic": is_toxic}

@action
async def check_output_toxicity(output_text: str = ""):
    """
    Check if the output text is toxic.
    """
    harmful_keywords = [
        "hack", "steal", "illegal", "bomb", "kill", "hurt", "attack",
        "exploit", "cheat", "fraud", "weapon", "violent", "abuse"
    ]
    
    output_lower = output_text.lower()
    is_toxic = any(keyword in output_lower for keyword in harmful_keywords)
    
    return {"is_toxic": is_toxic}
