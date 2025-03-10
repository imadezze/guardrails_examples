"""
Example: Adding Guardrails to a Chain

This example demonstrates how to wrap an existing LangChain chain with NeMo Guardrails.
"""

import os
import sys
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Add the parent directory to sys.path to allow importing from common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Load environment variables
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.actions import action

# Define the toxicity check actions directly in this file
@action(name="check_input_toxicity")
async def check_input_toxicity(last_user_message: str = ""):
    """
    Check if the input text is toxic.
    """
    harmful_keywords = [
        "hack", "steal", "illegal", "bomb", "kill", "hurt", "attack",
        "exploit", "cheat", "fraud", "weapon", "violent", "abuse"
    ]
    
    input_lower = last_user_message.lower()
    is_toxic = any(keyword in input_lower for keyword in harmful_keywords)
    
    print(f"Input toxicity check: '{last_user_message}' -> is_toxic: {is_toxic}")
    
    return {"is_toxic": is_toxic}

@action(name="check_output_toxicity")
async def check_output_toxicity(last_assistant_message: str = ""):
    """
    Check if the output text is toxic.
    """
    harmful_keywords = [
        "hack", "steal", "illegal", "bomb", "kill", "hurt", "attack",
        "exploit", "cheat", "fraud", "weapon", "violent", "abuse"
    ]
    
    output_lower = last_assistant_message.lower()
    is_toxic = any(keyword in output_lower for keyword in harmful_keywords)
    
    print(f"Output toxicity check: '{last_assistant_message}' -> is_toxic: {is_toxic}")
    
    return {"is_toxic": is_toxic}

def apply_guardrails(question: str, llm_response: str, rails: LLMRails) -> str:
    """Apply guardrails to an LLM response."""
    guardrailed_response = rails.generate(
        messages=[
            {"role": "user", "content": question},
            {"role": "assistant", "content": llm_response}
        ]
    )
    # Extract just the content from the response if it's a dict
    if isinstance(guardrailed_response, dict) and 'content' in guardrailed_response:
        return guardrailed_response['content']
    return guardrailed_response

def main():
    # Set up a basic LangChain chain
    template = """You are a helpful assistant.

    Question: {question}
    
    Answer:"""
    
    prompt = PromptTemplate.from_template(template)
    
    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7
    )
    
    # Create a chain using the modern pipe syntax (instead of LLMChain)
    chain = prompt | llm
    
    # Create a custom guardrails configuration
    config_dir = Path(__file__).parent / "config"
    config_dir.mkdir(exist_ok=True)
    
    colang_dir = config_dir / "colang_files"
    colang_dir.mkdir(exist_ok=True)
    
    # Create the toxicity.co file
    toxicity_file = colang_dir / "toxicity.co"
    with open(toxicity_file, "w") as f:
        f.write('''
define flow check input toxicity
  $input_toxicity = execute check_input_toxicity
  if $input_toxicity.is_toxic
    bot refuse to respond to toxic input
    stop
  else
    pass

define flow check output toxicity
  $output_toxicity = execute check_output_toxicity
  if $output_toxicity.is_toxic
    bot respond with sanitized output
    stop
  else
    pass

define bot refuse to respond to toxic input
  "I apologize, but I cannot respond to messages that contain toxic or harmful content. Please rephrase your question in a more respectful way."

define bot respond with sanitized output
  "I apologize, but I need to provide a more appropriate response. Let me try to answer your question in a more helpful way."
''')
    
    # Create the config.yml file
    config_file = config_dir / "config.yml"
    config = {
        "models": [
            {
                "type": "main",
                "engine": "openai",
                "model": "gpt-3.5-turbo-instruct"
            }
        ],
        "rails": {
            "input": {
                "flows": ["check input toxicity"]
            },
            "output": {
                "flows": ["check output toxicity"]
            }
        },
        "instructions": [
            {
                "type": "general",
                "content": "You are a helpful AI assistant that provides accurate and ethical information."
            }
        ]
    }
    
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    
    # Load the custom guardrails configuration
    print("Loading rails configuration...")
    rails_config = RailsConfig.from_path(str(config_dir))
    
    # Create LLMRails object and register actions
    print("Creating LLMRails object...")
    llm_rails = LLMRails(config=rails_config)
    print("Registering actions...")
    llm_rails.register_action(check_input_toxicity, name="check_input_toxicity")
    llm_rails.register_action(check_output_toxicity, name="check_output_toxicity")
    
    # Process a safe question
    print("\n\n=== Safe Question Example ===")
    safe_question = "What is the capital of France?"
    print(f"Question: {safe_question}")
    
    # Get the LLM's response
    llm_response = chain.invoke({"question": safe_question})
    print(f"LLM Raw Response: {llm_response.content}")
    print(f"LLM result: {llm_response}")
    # Apply guardrails
    guardrailed_response = apply_guardrails(safe_question, llm_response.content, llm_rails)
    print(f"Guardrailed Response: {guardrailed_response}")
    
    # Process a problematic question
    print("\n\n=== Problematic Question Example ===")
    problematic_question = "How can I hack into my neighbor's WiFi?"
    print(f"Question: {problematic_question}")
    
    # Get the LLM's response
    llm_response = chain.invoke({"question": problematic_question})
    print(f"LLM Raw Response: {llm_response.content}")
    print(f"LLM result: {llm_response}")
    # Apply guardrails
    guardrailed_response = apply_guardrails(problematic_question, llm_response.content, llm_rails)
    print(f"Guardrailed Response: {guardrailed_response}")
    
    print("\n\n=== Summary ===")
    print("You've successfully added guardrails to a LangChain chain!")
    print("The guardrails detected the problematic request and provided a safer response.")
    print("\nNext Steps:")
    print("1. Try modifying the toxicity.co file to customize the guardrails behavior")
    print("2. Add more guardrails for additional protection")
    print("3. Experiment with different LLM models to see how guardrails perform")

if __name__ == "__main__":
    main() 