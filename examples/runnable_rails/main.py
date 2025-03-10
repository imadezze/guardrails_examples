"""
Example: RunnableRails with Advanced Features

This example demonstrates how to use RunnableRails with advanced features like
key mapping, prompt passthrough, and streaming support.
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
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable import RunnableRails

def main():
    # Load the guardrails configuration
    config_path = Path(__file__).parent.parent / "common" / "config"
    config = RailsConfig.from_path(config_path)
    
    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7
    )
    
    # Create a prompt with multiple input fields
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that provides information about {topic}."),
        ("user", "{question}")
    ])
    
    # Create a chain that passes through all inputs
    chain = (
        {"topic": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    
    # 1. Basic RunnableRails with default configuration
    print("\n--- Example 1: Basic RunnableRails ---")
    
    # Wrap the chain with guardrails
    basic_rails = RunnableRails(config)
    guardrailed_chain = chain | basic_rails
    
    # Run the chain
    result = guardrailed_chain.invoke({
        "topic": "artificial intelligence",
        "question": "What is machine learning?"
    })
    print(f"Result: {result}")
    
    # 2. RunnableRails with key mapping
    print("\n--- Example 2: RunnableRails with Key Mapping ---")
    
    # Define a custom key mapping for inputs and outputs
    key_mapped_rails = RunnableRails(
        config,
        input_map={"messages": lambda inputs: [{"role": "user", "content": inputs["question"]}]},
        output_map=lambda outputs: {"response": outputs, "meta": {"passed_guardrails": True}}
    )
    
    # Create a chain with key mapping
    key_mapped_chain = chain | key_mapped_rails
    
    # Run the chain
    result = key_mapped_chain.invoke({
        "topic": "astronomy",
        "question": "What is a supernova?"
    })
    print(f"Result: {result}")
    
    # 3. RunnableRails with streaming support
    print("\n--- Example 3: RunnableRails with Streaming ---")
    
    # Note: We use streaming-compatible ChatOpenAI
    streaming_llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        streaming=True
    )
    
    streaming_chain = (
        {"topic": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | streaming_llm
    )
    
    # Wrap with guardrails that supports streaming
    streaming_rails = RunnableRails(config)
    streaming_guardrailed_chain = streaming_chain | streaming_rails
    
    print("Streaming response (chunk by chunk):")
    for chunk in streaming_guardrailed_chain.stream({
        "topic": "physics",
        "question": "What is quantum mechanics?"
    }):
        # In a real application, you would send these chunks to the client
        print(f"Chunk: {chunk}")
    
    # 4. RunnableRails with prompt passthrough
    print("\n--- Example 4: RunnableRails with Prompt Passthrough ---")
    
    # Create a chain with prompt passthrough
    passthrough_rails = RunnableRails(
        config,
        passthrough_inputs={"original_query": lambda x: x["question"]}
    )
    
    passthrough_chain = chain | passthrough_rails
    
    # Run the chain
    result = passthrough_chain.invoke({
        "topic": "biology",
        "question": "What is DNA?"
    })
    print(f"Result: {result}")

if __name__ == "__main__":
    main() 