"""
Example: LangSmith Integration with NeMo Guardrails

This example demonstrates how to use LangSmith for tracing, monitoring, and
debugging LangChain components that are wrapped with NeMo Guardrails.
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
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable import RunnableRails
from langsmith import Client
import langsmith

def main():
    # Check if LangSmith API key is set
    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        print("WARNING: LANGSMITH_API_KEY not set. LangSmith tracing will not work.")
        print("Set the LANGSMITH_API_KEY environment variable and try again.")
        return
    
    # Set up LangSmith project
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.environ.get("LANGSMITH_PROJECT", "nemo_guardrails_examples")
    
    # Set up a basic LangChain chain
    template = """You are a helpful AI assistant that provides information about space and astronomy.

    Question: {question}
    
    Answer:"""
    
    prompt = PromptTemplate.from_template(template)
    
    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # Create a chain
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Load the guardrails configuration
    config_path = Path(__file__).parent.parent / "common" / "config"
    config = RailsConfig.from_path(config_path)
    
    # Wrap the chain with guardrails
    guardrailed_chain = chain | RunnableRails(config)
    
    # Initialize LangSmith client
    client = Client()
    
    # Run a few examples with LangSmith tracing
    questions = [
        "What is a black hole?",
        "How far is the Moon from Earth?",
        "Is the Earth flat?",
        "How can I build a rocket at home?" # This might trigger safety rails
    ]
    
    print("\nSending questions to the guardrailed chain with LangSmith tracing enabled:")
    for question in questions:
        print(f"\nQuestion: {question}")
        with langsmith.trace(
            name="Guardrailed Astronomy Chain",
            project_name=os.environ["LANGCHAIN_PROJECT"]
        ) as trace:
            try:
                result = guardrailed_chain.invoke({"question": question})
                print(f"Response: {result}")
            except Exception as e:
                print(f"Error: {e}")
    
    print("\nTraces have been uploaded to LangSmith.")
    print(f"You can view them at: https://smith.langchain.com/o/me/projects/{os.environ['LANGCHAIN_PROJECT']}")

if __name__ == "__main__":
    main() 