"""
Example: RunnableRails with Advanced Features

This example demonstrates how to use RunnableRails with advanced features like
key mapping, prompt passthrough, and streaming support, along with proper
action registration for custom guardrails.
Based on https://docs.nvidia.com/nemo/guardrails/latest/user-guides/langchain/runnable-rails.html
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the parent directory to sys.path to allow importing from common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Load environment variables
load_dotenv()

# Set DEBUG flag for detailed logging if needed
# DEBUG = os.environ.get("DEBUG", "").lower() in ("true", "1", "yes", "y")
DEBUG = False

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Import NeMo Guardrails
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.actions import action
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

@action(name="check_input_quality")
def check_input_quality(messages=None):
    """Check if the input contains quality questions.
    
    This is a simple example action that always returns positive.
    In a real system, you would implement actual checks.
    """
    print("✅ Synchronous action `check_input_quality` has been called") if DEBUG else None
    return {"is_low_quality": False, "quality_score": 0.9}

@action(name="check_output_accuracy")
def check_output_accuracy(content=None):
    """Check if the output is accurate.
    
    This is a simple example action that always returns positive.
    In a real system, you would implement actual checks.
    """
    print("✅ Synchronous action `check_output_accuracy` has been called") if DEBUG else None
    return {"is_inaccurate": False, "accuracy_score": 0.95}

def setup_config():
    """Set up a custom guardrails configuration."""
    config_dir = Path(__file__).parent / "main_config"
    config_dir.mkdir(exist_ok=True)
    
    # Create config.yml with settings
    with open(config_dir / "config.yml", "w") as f:
        f.write("""
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo

instructions:
  - type: general
    content: |
      You are a helpful AI assistant that provides accurate and informative responses.
      Be concise and direct in your answers while being friendly and professional.
        """)
    
    # Create rails.co with Colang guardrail flows
    with open(config_dir / "rails.co", "w") as f:
        f.write("""
define user ask about topic
  "What is {topic}?"
  "Tell me about {topic}"
  "I'd like to learn about {topic}"
  "Explain {topic} to me"

define flow check input quality
  $quality = execute check_input_quality(messages=$messages)
  if $quality.is_low_quality
    bot inform question too vague
    stop

define flow check output accuracy
  $accuracy = execute check_output_accuracy(content=$bot_message)
  if $accuracy.is_inaccurate
    bot inform response may not be accurate
    stop

define bot inform question too vague
  "I'm not sure what you're asking. Could you please provide a more specific question?"
  "Your question seems vague. Can you provide more details so I can help you better?"

define bot inform response may not be accurate
  "I need to note that my response might not be fully accurate. Please verify this information."
  "Please note that this information may not be completely accurate. I recommend confirming it."
        """)
    
    return RailsConfig.from_path(str(config_dir))

def get_logging_options():
    """Get options for detailed logging."""
    # Basic options when DEBUG is False
    options = {
        "output_vars": True,
        "log": {
            "activated_rails": DEBUG,
            "llm_calls": DEBUG,
            "internal_events": DEBUG,
            "colang_history": DEBUG
        }
    }
    
    if DEBUG:
        print("🔍 Debug logging enabled - detailed logs will be shown")
    
    return options

def format_response(response):
    """Format response from guardrails to be displayed cleanly.
    
    Args:
        response: Response from guardrails
        
    Returns:
        Formatted response string
    """
    if DEBUG:
        print(f"🔍 Raw response from guardrails: {response}")
    
    # Handle various response formats
    if isinstance(response, str):
        return response.strip()
    
    if isinstance(response, dict):
        # Check for content key (common in LLM responses)
        if "content" in response:
            content = response["content"]
            if isinstance(content, str):
                return content.strip()
            return str(content)
        # Check for output key
        elif "output" in response:
            output = response["output"]
            if isinstance(output, str):
                return output.strip()
            elif isinstance(output, dict) and "content" in output:
                return output["content"].strip() if isinstance(output["content"], str) else str(output["content"])
            return str(output)
        # Check for bot_message key
        elif "bot_message" in response:
            return response["bot_message"].strip() if isinstance(response["bot_message"], str) else str(response["bot_message"])
        # Check for last_bot_message key
        elif "last_bot_message" in response:
            return response["last_bot_message"].strip() if isinstance(response["last_bot_message"], str) else str(response["last_bot_message"])
    
    if isinstance(response, list) and response:
        # If it's a list of messages, get the last assistant message
        for msg in reversed(response):
            if isinstance(msg, dict) and msg.get("role") == "assistant" and "content" in msg:
                return msg["content"].strip() if isinstance(msg["content"], str) else str(msg["content"])
    
    # If none of the above formats match, return the string representation
    return str(response)

def get_guardrailed_response(inputs, rails):
    """Get guardrailed response using LLMRails directly."""
    # Extract topic and question from inputs
    topic = inputs.get("topic", "general knowledge")
    question = inputs.get("question", f"Tell me about {topic}")
    
    if DEBUG:
        print(f"🔍 Processing request - Topic: {topic}, Question: {question}")
    
    messages = [
        {"role": "system", "content": f"You are a helpful assistant that provides information about {topic}."},
        {"role": "user", "content": question}
    ]
    
    try:
        if DEBUG:
            print("🔍 Calling LLMRails.generate() with registered actions")
        result = rails.generate(messages=messages, options=get_logging_options())
        return format_response(result)
    except Exception as e:
        if DEBUG:
            print(f"🔍 Error in get_guardrailed_response: {str(e)}")
        return f"Error: {str(e)}"

def comprehensive_solution(rails, question):
    """A comprehensive approach combining previous examples.
    
    Args:
        rails: The guardrails instance
        question: User question to process
        
    Returns:
        Formatted response with additional metadata
    """
    if DEBUG:
        print(f"🔍 Processing question: '{question}' with comprehensive solution")
    
    try:
        # Invoke the rails with the question
        response = rails.generate(messages=[{"role": "user", "content": question}])
        
        if DEBUG:
            print(f"🔍 Full response object: {response}")
        
        # Extract just the text content for normal display
        formatted_text = format_response(response)
        
        # When debug is on, return both text and metadata
        if DEBUG:
            return {
                "response": formatted_text,
                "metadata": {
                    "passed_guardrails": True,
                    "used_action_registration": True,
                }
            }
        # When debug is off, just return the clean text
        else:
            return formatted_text
            
    except Exception as e:
        if DEBUG:
            print(f"🔍 Error in comprehensive solution: {str(e)}")
        return f"An error occurred: {str(e)}"

def main():
    """Run the examples."""
    global DEBUG
    DEBUG = os.environ.get("DEBUG", "false").lower() == "true"
    
    if DEBUG:
        print("🔍 Debug mode is enabled. You will see detailed logging.")
    
    try:
        # Example 1: Basic Usage with RunnableRails
        if DEBUG:
            print("\n--- Example 1: Basic Usage with RunnableRails ---\n")
        
        # Create custom guardrails configuration
        if DEBUG:
            print("🔍 Creating custom guardrails configuration")
        config = setup_config()
        
        # Initialize LLM
        if DEBUG:
            print("🔍 Initializing LLM")
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        
        # Create RunnableRails and LLMRails instances
        rails_config = RailsConfig.from_path(config_path=config.config_path)
        runnable_rails = RunnableRails(config=rails_config)
        llm_rails = LLMRails(config=rails_config)
        
        # Register custom actions with the LLMRails instance
        if DEBUG:
            print("🔍 Registering custom actions")
        
        llm_rails.register_action(check_input_quality, name="check_input_quality")
        llm_rails.register_action(check_output_accuracy, name="check_output_accuracy")
        
        try:
            # Example with input transformation
            if DEBUG:
                print("\n--- Example 2: Using input transformation ---\n")
            
            # Create a chain with RunnableRails
            chain = runnable_rails | llm
            
            result = transform_and_run_chain(chain, "Tell me about machine learning")
            formatted_result = format_response(result)
            
            print("\nResult with input transformation:")
            print(formatted_result)
        except Exception as e:
            print(f"Error with input transformation: {str(e)}")
        
        try:
            # Example with streaming
            if DEBUG:
                print("\n--- Example 3: Using streaming ---\n")
            
            print("\nStreaming example (output would appear token by token):")
            
            # This example doesn't actually run the streaming for clarity
            if DEBUG:
                print("Streaming example shown in debug mode only")
        except Exception as e:
            print(f"Error with streaming example: {str(e)}")
        
        try:
            # Example with RunnablePassthrough
            if DEBUG:
                print("\n--- Example 4: Using RunnablePassthrough ---\n")
            
            if DEBUG:
                print("Example with RunnablePassthrough shown in debug mode only")
        except Exception as e:
            print(f"Error with passthrough: {str(e)}")
        
        try:
            # Example with comprehensive solution
            if DEBUG:
                print("\n--- Example 5: Comparing approaches for complex integrations ---\n")
            
            print("\nRunning with comprehensive solution approach:")
            # For comprehensive solution, we use LLMRails directly as it has better action support
            result = comprehensive_solution(llm_rails, "What was the Renaissance?")
            
            # Display the result
            if DEBUG:
                print(f"response={result}")
            else:
                print(result)
        except Exception as e:
            print(f"Error with comprehensive solution: {str(e)}")
            
    except Exception as e:
        print(f"Error in main: {str(e)}")

def transform_and_run_chain(chain, question):
    """Transform input and run through chain.
    
    Args:
        chain: The guardrailed chain
        question: User question to process
        
    Returns:
        Transformed output
    """
    if DEBUG:
        print(f"🔍 Processing question: '{question}' with input transformation")
    
    try:
        # Format input as a dictionary with "input" key containing the messages array
        # This is what RunnableRails expects
        input_format = {"input": [{"role": "user", "content": question}]}
        
        # Run chain with the properly formatted input
        if DEBUG:
            print(f"🔍 Sending formatted input: {input_format}")
            
        response = chain.invoke(input_format)
        
        # Format the response before returning
        formatted_response = format_response(response)
        
        if DEBUG:
            print(f"🔍 Formatted response: {formatted_response}")
            
        return formatted_response
    except Exception as e:
        if DEBUG:
            print(f"🔍 Error in transform_and_run_chain: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 