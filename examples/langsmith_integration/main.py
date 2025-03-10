"""
Example: LangSmith Integration with NeMo Guardrails

This example demonstrates how to use LangSmith for tracing, monitoring, and
debugging LangChain components that are wrapped with NeMo Guardrails.

Note: This approach uses direct integration with LLMRails instead of RunnableRails
to ensure proper action registration and execution within the LangChain tracing system.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the parent directory to sys.path to allow importing from common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Load environment variables
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.actions import action
from langsmith import Client
import langsmith

# Define required actions for guardrails
@action(name="check_input_toxicity")
def check_input_toxicity(messages=None):
    """Check if input contains toxic content.
    
    This action is called by the guardrails system before processing user input.
    
    Args:
        messages: List of message objects from the user
        
    Returns:
        Dict with is_toxic flag and toxicity_score
    """
    # Simple implementation - in production would use a real toxicity detector
    toxic_terms = ["kill", "hate", "destroy", "stupid", "idiot"]
    
    if messages and len(messages) > 0:
        last_message = messages[-1]["content"].lower()
        for term in toxic_terms:
            if term in last_message:
                return {"is_toxic": True, "toxicity_score": 0.9}
    
    return {"is_toxic": False, "toxicity_score": 0.1}

@action(name="check_output_toxicity")
def check_output_toxicity(bot_response=None):
    """Check if output contains toxic content.
    
    This action is called by the guardrails system before returning the LLM's response.
    
    Args:
        bot_response: Response from the bot
        
    Returns:
        Dict with is_toxic flag and toxicity_score
    """
    # Simple implementation - in production would use a real toxicity detector
    toxic_terms = ["kill", "hate", "destroy", "classified", "illegal"]
    
    if bot_response:
        bot_text = bot_response.lower()
        for term in toxic_terms:
            if term in bot_text:
                return {"is_toxic": True, "toxicity_score": 0.9}
    
    return {"is_toxic": False, "toxicity_score": 0.1}

def main():
    # Check if LangSmith API key is set
    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        print("WARNING: LANGSMITH_API_KEY not set. LangSmith tracing will not work.")
        print("Set the LANGSMITH_API_KEY environment variable and try again.")
        print("Continuing without LangSmith tracing...")
    else:
        # Set up LangSmith project
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = os.environ.get("LANGSMITH_PROJECT", "nemo_guardrails_examples")
    
    # Set up a basic LangChain prompt
    template = """You are a helpful AI assistant that provides information about space and astronomy.

    Question: {question}
    
    Answer:"""
    
    prompt = PromptTemplate.from_template(template)
    
    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # Load the guardrails configuration
    config_path = Path(__file__).parent.parent / "common" / "config"
    config = RailsConfig.from_path(str(config_path))
    
    # Create an LLMRails instance - this is the core guardrails engine
    rails = LLMRails(config)
    
    # Register the required actions - crucial for the guardrails to work properly
    rails.register_action(check_input_toxicity)
    rails.register_action(check_output_toxicity)
    
    # Helper function to process requests through guardrails
    # This approach allows for clean integration with LangSmith tracing
    def process_with_guardrails(inputs):
        """Process user queries through the guardrails system.
        
        This function:
        1. Extracts the question from inputs
        2. Formats it with the prompt template 
        3. Passes it through the guardrails system
        4. Returns the guardrailed response
        
        Args:
            inputs: Dictionary with 'question' key
            
        Returns:
            Guardrailed response from the LLM
        """
        # Extract the question
        question = inputs.get("question", "")
        
        # Format the prompt with the question
        formatted_prompt = template.format(question=question)
        
        # Process through guardrails
        messages = [{"role": "user", "content": formatted_prompt}]
        try:
            # This is where the guardrails magic happens
            # The rails.generate method applies all input/output guardrails
            response = rails.generate(messages=messages)
            
            # Extract content from the response
            if isinstance(response, dict) and "content" in response:
                return response["content"]
            return str(response)
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Create a LangChain runnable from our processing function
    # This allows integration with LangSmith tracing
    chain = RunnableLambda(process_with_guardrails)
    
    # Sample questions to test the guardrails
    questions = [
        "What is a black hole?",
        "How far is the Moon from Earth?",
        "Is the Earth flat?",  # This might trigger some guardrails
        "How can I build a rocket at home?"  # This might trigger safety rails
    ]
    
    print("\nSending questions to the guardrailed chain with LangSmith tracing enabled:")
    for question in questions:
        print(f"\nQuestion: {question}")
        # Use LangSmith tracing to capture the execution
        with langsmith.trace(
            name="Guardrailed Astronomy Chain",
            project_name=os.environ.get("LANGCHAIN_PROJECT", "nemo_guardrails_examples")
        ) as trace:
            try:
                # Invoke the chain with the question
                result = chain.invoke({"question": question})
                print(f"Response: {result}")
            except Exception as e:
                print(f"Error: {str(e)}")
    
    print("\nTraces have been uploaded to LangSmith.")
    print(f"You can view them at: https://smith.langchain.com/o/me/projects/{os.environ.get('LANGCHAIN_PROJECT', 'nemo_guardrails_examples')}")
    print("\nKey Insights from This Example:")
    print("1. Custom actions must be properly registered with LLMRails to work")
    print("2. LangSmith tracing captures all steps in the guardrailed generation process")
    print("3. The approach demonstrated bridges LangChain's component system with NeMo Guardrails")
    print("4. Tracing provides valuable insights into guardrails activation and behavior")

if __name__ == "__main__":
    main() 