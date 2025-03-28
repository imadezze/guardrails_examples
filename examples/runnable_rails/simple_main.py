"""
Simple Example: NeMo Guardrails Integration with LangChain

This example demonstrates how to integrate NeMo Guardrails with LangChain
using the LLMRails API directly for content filtering.

Based on: https://docs.nvidia.com/nemo/guardrails/latest/user-guides/langchain/runnable-rails.html
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

# Import the required packages
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.actions import action
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

# Set to True to enable detailed logging
DEBUG = False

# Configure logging options based on debug flag
def get_logging_options():
    """Return logging options based on debug flag."""
    if not DEBUG:
        return {}
    
    return {
        "output_vars": True,
        "log": {
            "activated_rails": True,
            "llm_calls": True,
            "internal_events": True
        }
    }

# Format response for nice display
def format_response(response):
    """Format a response for nice display."""
    if isinstance(response, dict) and 'content' in response:
        return response['content']
    elif isinstance(response, list) and response and isinstance(response[0], dict) and 'content' in response[0]:
        return response[0]['content']
    return str(response)

# Define the toxicity check actions with the proper names matching the colang flow
@action(name="check_input_toxicity")
def check_input_toxicity(messages=None):
    """Check if the input is toxic, exactly matching the action name used in the flow."""
    if not messages or not isinstance(messages, list) or not messages:
        return {"is_toxic": False, "toxicity_score": 0.0}
    
    # Get the last message content
    text = messages[-1].get("content", "").lower() if isinstance(messages[-1], dict) else ""
    
    # Check for potentially problematic terms
    problematic_terms = ["dangerous weapons", "illegal", "harmful", "hack", "exploit"]
    is_toxic = any(term in text for term in problematic_terms)
    
    if is_toxic and DEBUG:
        print(f"DEBUG: Input toxicity detected: {text}")
    
    return {
        "is_toxic": is_toxic,
        "toxicity_score": 0.9 if is_toxic else 0.1
    }

@action(name="check_output_toxicity")
def check_output_toxicity(content=None):
    """Check if the output is toxic, exactly matching the action name used in the flow."""
    if not content or not isinstance(content, str):
        return {"is_toxic": False, "toxicity_score": 0.0}
    
    content = content.lower()
    
    # Check for potentially problematic terms
    problematic_terms = ["dangerous weapons", "illegal", "harmful", "hack", "exploit"]
    is_toxic = any(term in content for term in problematic_terms)
    
    if is_toxic and DEBUG:
        print(f"DEBUG: Output toxicity detected: {content}")
    
    return {
        "is_toxic": is_toxic,
        "toxicity_score": 0.9 if is_toxic else 0.1
    }

def setup_config():
    """Set up and return the configuration for NeMo Guardrails."""
    # Create a directory for our custom configuration
    config_dir = Path(__file__).parent / "simple_config"
    config_dir.mkdir(exist_ok=True)
    
    # Create Colang files directory
    colang_dir = config_dir / "colang_files"
    colang_dir.mkdir(exist_ok=True)
    
    # Define Colang flows for toxicity checks
    with open(colang_dir / "toxicity.co", "w") as f:
        f.write("""
define flow check input toxicity
  $toxicity = execute check_input_toxicity(messages=$messages)
  if $toxicity.is_toxic
    bot refuse to respond due to harmful content
    stop

define flow check output toxicity
  $toxicity = execute check_output_toxicity(content=$generation)
  if $toxicity.is_toxic
    bot provide compliant response
    stop

define bot refuse to respond due to harmful content
  "I apologize, but I cannot respond to content related to harmful topics."

define bot provide compliant response
  "I apologize, but I cannot provide the information you're looking for as it may contain harmful content."
""")
    
    # Create a minimal config.yml
    with open(config_dir / "config.yml", "w") as f:
        f.write("""
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo

instructions:
  - type: general
    content: |
      You are a helpful AI assistant that provides appropriate and safe information.
      When asked for a joke, make sure it is appropriate for all audiences.

rails:
  input:
    flows:
      - check input toxicity
  output:
    flows:
      - check output toxicity
""")
    
    # Create and return the config
    return RailsConfig.from_path(str(config_dir))

def test_direct_llm_rails(rails):
    """Test using LLMRails directly for message generation."""
    print("\n" + "=" * 50)
    print(" APPROACH 1: Direct usage with LLMRails ".center(50, "="))
    print("=" * 50)
    
    # Test with safe messages
    safe_messages = [
        {"role": "user", "content": "Tell me a joke about programming"}
    ]
    
    problematic_messages = [
        {"role": "user", "content": "Tell me a joke about dangerous weapons"}
    ]
    
    # Get logging options
    logging_options = get_logging_options()
    
    try:
        print("\n📝 Safe topic:")
        print("-" * 50)
        print("User: Tell me a joke about programming")
        result = rails.generate(messages=safe_messages, options=logging_options)
        formatted_response = format_response(result)
        print(f"Assistant: {formatted_response}")
        
        # Print activated rails if in debug mode
        if DEBUG and hasattr(result, "log") and hasattr(result.log, "activated_rails"):
            print("\n🔍 Activated Rails:")
            for rail in result.log.activated_rails:
                print(f"- {rail.type}: {rail.name}")
                if hasattr(rail, "decisions"):
                    print(f"  Decisions: {rail.decisions}")
                if hasattr(rail, "executed_actions") and rail.executed_actions:
                    print("  Executed Actions:")
                    for action in rail.executed_actions:
                        print(f"    - {action.action_name}")
    except Exception as e:
        print(f"❌ Error with safe topic: {str(e)}")
    
    try:
        print("\n📝 Problematic topic:")
        print("-" * 50)
        print("User: Tell me a joke about dangerous weapons")
        result = rails.generate(messages=problematic_messages, options=logging_options)
        formatted_response = format_response(result)
        print(f"Assistant: {formatted_response}")
        
        # Print activated rails if in debug mode
        if DEBUG and hasattr(result, "log") and hasattr(result.log, "activated_rails"):
            print("\n🔍 Activated Rails:")
            for rail in result.log.activated_rails:
                print(f"- {rail.type}: {rail.name}")
                if hasattr(rail, "decisions"):
                    print(f"  Decisions: {rail.decisions}")
    except Exception as e:
        print(f"❌ Error with problematic topic: {str(e)}")

def test_langchain_integration(custom_config):
    """Test LangChain integration with and without guardrails."""
    print("\n" + "=" * 50)
    print(" APPROACH 2: Using LangChain with RunnableRails ".center(50, "="))
    print("=" * 50)
    
    # Create a simple LangChain example
    prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
    model = ChatOpenAI()
    output_parser = StrOutputParser()
    
    # Create a standard chain (without guardrails)
    standard_chain = prompt | model | output_parser
    
    # Create a RunnableRails instance
    runnable_guardrails = RunnableRails(config=custom_config)
    
    # NOTE: Unlike LLMRails, RunnableRails doesn't provide a direct way to register actions
    # This is a limitation of the current RunnableRails implementation
    print("⚠️  Note: RunnableRails may have limitations with action registration.")
    print("    Using fallback to standard responses for demonstration purposes.")
    
    # Create a guardrailed chain - note the parentheses as specified in the docs
    guardrailed_chain = prompt | (runnable_guardrails | model) | output_parser
    
    # Run standard chain with safe topic
    print("\n📝 Standard LangChain (no guardrails) - safe topic:")
    print("-" * 50)
    print("User: Tell me a joke about programming")
    try:
        result = standard_chain.invoke({"topic": "programming"})
        print(f"Assistant: {result}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Run standard chain with problematic topic  
    print("\n📝 Standard LangChain (no guardrails) - problematic topic:")
    print("-" * 50)
    print("User: Tell me a joke about dangerous weapons")
    try:
        result = standard_chain.invoke({"topic": "dangerous weapons"})
        print(f"Assistant: {result}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Create a workaround function that uses LLMRails directly but with LangChain's format
    def get_guardrailed_response(topic):
        """Get guardrailed response using LLMRails directly."""
        # Create a fresh rails instance
        rails = LLMRails(custom_config)
        # Register the actions
        rails.register_action(check_input_toxicity)
        rails.register_action(check_output_toxicity)
        
        messages = [
            {"role": "user", "content": f"Tell me a joke about {topic}"}
        ]
        try:
            result = rails.generate(messages=messages, options=get_logging_options())
            return format_response(result)
        except Exception as e:
            return f"Error: {str(e)}"
        
    # Use LLMRails (which works correctly) to demonstrate what RunnableRails should do
    print("\n📝 Workaround: Using LLMRails for guardrailed responses - safe topic:")
    print("-" * 50)
    print("User: Tell me a joke about programming")
    result = get_guardrailed_response("programming")
    print(f"Assistant: {result}")
    
    print("\n📝 Workaround: Using LLMRails for guardrailed responses - problematic topic:")
    print("-" * 50)
    print("User: Tell me a joke about dangerous weapons")
    result = get_guardrailed_response("dangerous weapons")
    print(f"Assistant: {result}")
    
    # Still attempt to run RunnableRails for completeness
    print("\n📝 RunnableRails (likely to show action not found) - safe topic:")
    print("-" * 50)
    print("User: Tell me a joke about programming")
    try:
        result = guardrailed_chain.invoke({"topic": "programming"})
        print(f"Assistant: {result}")
    except Exception as e:
        print(f"❌ Error with guardrailed chain: {str(e)}")
    
    print("\n📝 RunnableRails (likely to show action not found) - problematic topic:")
    print("-" * 50)
    print("User: Tell me a joke about dangerous weapons")
    try:
        result = guardrailed_chain.invoke({"topic": "dangerous weapons"})
        print(f"Assistant: {result}")
    except Exception as e:
        print(f"❌ Error with guardrailed chain: {str(e)}")

def main():
    """Main function to demonstrate NeMo Guardrails with LangChain."""
    print("\n" + "=" * 50)
    print(" Simple NeMo Guardrails Example ".center(50, "="))
    print("=" * 50)
    print(f"Debug mode: {'✅ Enabled' if DEBUG else '❌ Disabled'}")
    
    # Setup configuration
    custom_config = setup_config()
    
    # Create LLMRails instance for direct message handling
    rails = LLMRails(custom_config)
    
    # Explicitly register the actions
    rails.register_action(check_input_toxicity)
    rails.register_action(check_output_toxicity)
    
    # Test direct usage with LLMRails
    test_direct_llm_rails(rails)
    
    # Test LangChain integration
    test_langchain_integration(custom_config)

if __name__ == "__main__":
    # Enable debug mode by setting environment variable
    if os.environ.get("DEBUG", "").lower() in ("true", "1", "yes", "y"):
        DEBUG = True
    
    main() 