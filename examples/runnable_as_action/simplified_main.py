"""
Simplified example of using Runnable-As-Action with NeMo Guardrails.
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
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions import action

def main():
    # Set up configuration directories
    config_dir = Path(__file__).parent / "simplified_config"
    config_dir.mkdir(exist_ok=True)
    colang_dir = config_dir / "colang_files"
    colang_dir.mkdir(exist_ok=True)
    
    # Set up a custom colang file for this example
    custom_colang = """
    define user ask for calculation
      "Calculate {expression}"
      "What is {expression}?"
      "Compute {expression}"
    
    define flow calculation request
      user ask for calculation
      $result = execute extract_and_calculate(user_message=$user_message)
      bot provide calculation result(result=$result)
    
    define bot provide calculation result
      "The result is {result}."
    """
    
    # Write the custom colang to a file
    with open(colang_dir / "calculator.co", "w") as f:
        f.write(custom_colang)
    
    # Create a basic config.yml for this example
    config = {
        "models": [
            {
                "type": "main",
                "engine": "openai",
                "model": "gpt-3.5-turbo-instruct"
            }
        ],
        "rails": {
            "dialog": {
                "flows": ["calculation request"]
            }
        },
        "instructions": [
            {
                "type": "general",
                "content": "You are a helpful AI assistant that can perform calculations. When asked about calculations, you'll compute the result."
            }
        ]
    }
    
    with open(config_dir / "config.yml", "w") as f:
        yaml.dump(config, f)
    
    # Create the calculator chain with LangChain
    calculator_template = """You are a math calculator.
    Calculate the following expression and return ONLY the numeric result without explanation or text.
    
    Expression: {expression}
    
    Result: """
    
    calculator_prompt = PromptTemplate.from_template(calculator_template)
    
    calculator_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )
    
    # Use modern pipe syntax
    calculator_chain = calculator_prompt | calculator_llm
    
    # Load the config
    config = RailsConfig.from_path(str(config_dir))
    
    # Initialize the guardrails
    rails = LLMRails(config)
    
    # Define the calculator function
    def calculate_expression(expression: str) -> str:
        """Calculate a mathematical expression."""
        if os.environ.get("DEBUG", "").lower() == "true":
            print(f"DEBUG: Calculating expression: {expression!r}")
            print(f"DEBUG: Expression type: {type(expression)}")
        
        # Check if the expression is still a variable placeholder
        if expression == "$expression":
            if os.environ.get("DEBUG", "").lower() == "true":
                print("WARNING: The expression variable wasn't properly extracted from the user input")
            return "I'm sorry, but it seems that the expression you want me to calculate is missing. Please provide the expression, and I'll calculate the result for you."
            
        # Now try to calculate with the provided expression
        try:
            response = calculator_chain.invoke({"expression": expression})
            # Handle the new response format (AIMessage)
            result = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            if os.environ.get("DEBUG", "").lower() == "true":
                print(f"DEBUG: Calculation result: {result}")
            return result
        except Exception as e:
            if os.environ.get("DEBUG", "").lower() == "true":
                print(f"ERROR: Failed to calculate expression: {str(e)}")
            return f"I'm sorry, I couldn't calculate '{expression}'. Please check if the expression is valid."
    
    # Register the action with rails
    rails.register_action(calculate_expression, name="calculate_expression")
    
    # Create a wrapper function to extract the expression directly from the user message
    def extract_and_calculate(user_message: str) -> str:
        """Extract the expression from the user message and calculate it."""
        import re
        
        # Try to extract the expression using regex
        calculate_pattern = r"Calculate\s+(.*)"
        compute_pattern = r"Compute\s+(.*)"
        what_is_pattern = r"What is\s+(.*)\??"
        
        match = re.search(calculate_pattern, user_message, re.IGNORECASE)
        if not match:
            match = re.search(compute_pattern, user_message, re.IGNORECASE)
        if not match:
            match = re.search(what_is_pattern, user_message, re.IGNORECASE)
            
        if match:
            expression = match.group(1).strip()
            if os.environ.get("DEBUG", "").lower() == "true":
                print(f"DEBUG: Extracted expression: {expression!r}")
            return calculate_expression(expression)
        else:
            return "I couldn't understand the expression. Please provide it in the format 'Calculate X'."
    
    # Register the extraction wrapper
    rails.register_action(extract_and_calculate, name="extract_and_calculate")
    
    # Test the calculation flow
    print("\n--- Testing Calculator Flow ---\n")
    test_messages = [
        "Calculate 23 + 45",
        "Calculate sqrt(144) + 10",
        "Calculate (35 * 12) / 7"
    ]
    
    for user_message in test_messages:
        print(f"User: {user_message}")
        if os.environ.get("DEBUG", "").lower() == "true":
            print(f"DEBUG: Raw message: {user_message}")
        
        # Generate the response
        # Temporarily redirect stderr to suppress framework messages
        import io
        
        if os.environ.get("DEBUG", "").lower() != "true":
            # Only suppress messages in non-debug mode
            original_stderr = sys.stderr
            sys.stderr = io.StringIO()
            
        try:
            response = rails.generate(
                messages=[{
                    "role": "user", 
                    "content": user_message
                }],
                options={
                    "output_vars": True,
                    "log": {
                        "activated_rails": True,
                        "llm_calls": True,
                        "internal_events": True
                    }
                }
            )
        finally:
            # Restore stderr
            if os.environ.get("DEBUG", "").lower() != "true":
                sys.stderr = original_stderr
        
        # Format and display a clean response
        if hasattr(response, "output_data") and "result" in response.output_data:
            result = response.output_data["result"]
            print(f"\nAssistant: The result of the calculation is {result}\n")
        else:
            print(f"\nAssistant: {response[0]['content'] if isinstance(response, list) else response}\n")
        
        # Only show debugging information if DEBUG flag is set
        if os.environ.get("DEBUG", "").lower() == "true":
            # Print detailed logs if available
            if hasattr(response, "log") and hasattr(response.log, "activated_rails"):
                print("\nActivated Rails:")
                for rail in response.log.activated_rails:
                    print(f"- {rail.type}: {rail.name}")
                    print(f"  Decisions: {rail.decisions}")
                    if hasattr(rail, "executed_actions") and rail.executed_actions:
                        print("  Executed Actions:")
                        for action in rail.executed_actions:
                            print(f"    - {action.action_name} with params: {action.action_params}")
                            
            # Print output vars if available
            if hasattr(response, "output_data"):
                print("\nOutput Data:")
                for key, value in response.output_data.items():
                    print(f"- {key}: {value}")


if __name__ == "__main__":
    main() 