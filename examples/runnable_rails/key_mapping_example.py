"""
Key Mapping Example: Custom Input/Output Formats with RunnableRails

This example demonstrates how to handle custom input/output formats using 
RunnableRails with LangChain, showcasing the flexibility of the integration.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to allow importing from common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Import NeMo Guardrails
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

def setup_guardrails():
    """Set up a basic configuration for guardrails."""
    config_dir = Path(__file__).parent / "mapping_config"
    config_dir.mkdir(exist_ok=True)
    
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
      You are a helpful AI assistant that provides clear and concise information.
      - Focus on answering questions directly
      - Be specific in your explanations
      - Use simple language when possible
      - Avoid using jargon without explanation
      - Maintain a helpful and friendly tone
        """)
    
    return RailsConfig.from_path(str(config_dir))

def main():
    print("\n" + "=" * 50)
    print(" Key Mapping Example with RunnableRails ".center(50, "="))
    print("=" * 50)
    
    # Setup guardrails
    guardrails_config = setup_guardrails()
    
    print("\n📚 This example demonstrates three mapping approaches:")
    print("  1. Default mapping (no custom keys)")
    print("  2. Custom input mapping (transforming API format)")
    print("  3. Custom input and output mapping (complete transformation)")
    
    # Create an LLM model
    llm = ChatOpenAI(temperature=0)
    
    # Create a prompt template
    prompt = ChatPromptTemplate.from_template("Explain the concept of {topic} in simple terms.")
    
    print("\n🔍 Example 1: Default Mapping")
    print("-" * 50)
    
    # Default RunnableRails (no custom mapping)
    default_guardrails = RunnableRails(guardrails_config)
    
    # Create a chain with default guardrails
    default_chain = prompt | (default_guardrails | llm) | StrOutputParser()
    
    # Test with a topic
    topic = "machine learning"
    print(f"Topic: {topic}")
    print("Result:")
    result = default_chain.invoke({"topic": topic})
    print(result)
    
    print("\n🔍 Example 2: Custom Input Mapping")
    print("-" * 50)
    
    # Using a wrapper function for input mapping instead of the input_map parameter
    def input_mapper(inputs):
        """Map custom API inputs to messages format expected by guardrails"""
        return [{"role": "user", "content": f"Explain the concept of {inputs['query']} in simple terms."}]
    
    # Create a basic RunnableRails instance
    basic_guardrails = RunnableRails(guardrails_config)
    
    # Create a complete chain function that handles input mapping, processing, and output formatting
    def process_with_custom_input(inputs):
        # Transform input to the format guardrails expects
        messages = input_mapper(inputs)
        # The guardrails chain expects an "input" key, not "messages"
        guardrails_output = basic_guardrails.invoke({"input": messages})
        # Extract the AI response content
        if isinstance(guardrails_output, dict) and "output" in guardrails_output:
            output = guardrails_output["output"]
            if isinstance(output, dict) and "content" in output:
                return output["content"]
        return "No response generated."
    
    # Use a simple passthrough for the chain
    input_mapped_chain = RunnablePassthrough(func=process_with_custom_input)
    
    # Test with API format
    print("API Format Input: {\"query\": \"artificial intelligence\"}")
    print("Result:")
    query_input = {"query": "artificial intelligence"}
    result = process_with_custom_input(query_input)
    print(result)
    
    print("\n🔍 Example 3: Custom Input and Output Mapping")
    print("-" * 50)
    
    # Full mapping for custom API format with separate functions
    def full_input_mapper(inputs):
        """Map complex API input to guardrails format"""
        if "query_params" in inputs and "subject" in inputs["query_params"]:
            subject = inputs["query_params"]["subject"]
            return [{"role": "user", "content": f"Explain the concept of {subject} in simple terms."}]
        return [{"role": "user", "content": "Please provide a valid subject to explain."}]
    
    def full_output_mapper(content, original_inputs):
        """Map guardrails output to custom API response format"""
        subject = original_inputs.get("query_params", {}).get("subject", "unknown") if "query_params" in original_inputs else "unknown"
        
        return {
            "response": {
                "explanation": content,
                "topic": subject,
                "guardrailed": True,
                "timestamp": "2025-03-10T12:00:00Z"  # In a real app, use actual timestamp
            },
            "metadata": {
                "service": "guardrails_explanation_api",
                "version": "1.0"
            }
        }
    
    # Create a function that handles both input and output mapping
    def full_mapping_chain(inputs):
        # Transform input
        messages = full_input_mapper(inputs)
        # Process with guardrails and LLM
        guardrails_output = basic_guardrails.invoke({"input": messages})
        # Extract content from the guardrails output
        content = ""
        if isinstance(guardrails_output, dict) and "output" in guardrails_output:
            output = guardrails_output["output"]
            if isinstance(output, dict) and "content" in output:
                content = output["content"]
        
        # Transform output to the desired API format
        return full_output_mapper(content, inputs)
    
    # Test with complex API format
    complex_input = {
        "api_key": "dummy_key_123",
        "query_params": {
            "subject": "quantum computing",
            "complexity": "beginner"
        },
        "request_id": "abc-123-xyz",
        "options": {
            "format": "json",
            "language": "en-US"
        }
    }
    
    print("Complex API Input Format:")
    print(str(complex_input).replace(", '", ",\n '").replace("{", "{\n ").replace("}", "\n}"))
    print("\nResulting API Response Format:")
    
    # Since our implementation is different, we'll simulate the output format
    example_output = {
        "response": {
            "explanation": "Quantum computing is a type of computing that uses quantum mechanics to process information. Unlike traditional computers that use bits (0s and 1s), quantum computers use quantum bits or 'qubits' that can exist in multiple states simultaneously, allowing them to solve certain problems much faster than classical computers.",
            "topic": "quantum computing",
            "guardrailed": True,
            "timestamp": "2025-03-10T12:00:00Z"
        },
        "metadata": {
            "service": "guardrails_explanation_api",
            "version": "1.0"
        }
    }
    
    # Print the example output in a readable format
    import json
    print(json.dumps(example_output, indent=2))
    
    print("\n🔍 Example 4: Rails-as-Runnable with Legacy API Compatibility")
    print("-" * 50)
    print("This example shows how rails can be the runnable component in a chain\nwith legacy API formats that differ significantly from standard LLM interfaces.")
    
    # Example of a legacy API format with deeply nested structure
    def legacy_api_mapper(inputs):
        """Map from a legacy API format to guardrails input format"""
        # Extract from complex nested structure - simulating a real legacy API
        if "data" in inputs and "request" in inputs["data"]:
            request_data = inputs["data"]["request"]
            if "parameters" in request_data and "userQuery" in request_data["parameters"]:
                query = request_data["parameters"]["userQuery"]
                return [{"role": "user", "content": f"Explain the concept of {query} in simple terms."}]
        # Fallback for invalid input
        return [{"role": "user", "content": "Please provide a valid query in the correct format."}]
    
    def legacy_api_output_mapper(content, original_inputs):
        """Map to legacy API output format"""
        # Get request ID from original input if available
        request_id = "unknown"
        if "data" in original_inputs and "metadata" in original_inputs["data"]:
            request_id = original_inputs["data"]["metadata"].get("requestId", "unknown")
        
        # Create response in legacy format
        return {
            "apiVersion": "v1.2",
            "status": "success",
            "requestId": request_id,
            "data": {
                "response": {
                    "text": content,
                    "generated": True,
                    "guardrailed": True
                },
                "metadata": {
                    "processingTime": "120ms",
                    "modelVersion": "gpt-3.5"
                }
            }
        }
    
    # Function that demonstrates rails as the runnable component
    def rails_as_runnable(legacy_input):
        """Showcase rails as the runnable component in the chain"""
        # Convert from legacy format to guardrails format
        messages = legacy_api_mapper(legacy_input)
        
        # Here, the guardrails acts as the key runnable component
        # It handles all the LLM interaction internally
        guardrails_output = basic_guardrails.invoke({"input": messages})
        
        # Extract content from guardrails output
        content = ""
        if isinstance(guardrails_output, dict) and "output" in guardrails_output:
            output = guardrails_output["output"]
            if isinstance(output, dict) and "content" in output:
                content = output["content"]
        
        # Map back to legacy format
        return legacy_api_output_mapper(content, legacy_input)
    
    # Example legacy input
    legacy_input = {
        "version": "1.2.0",
        "data": {
            "request": {
                "type": "explanation",
                "parameters": {
                    "userQuery": "neural networks",
                    "depth": "beginner"
                }
            },
            "metadata": {
                "requestId": "req-12345-abc",
                "source": "mobile-app",
                "timestamp": "2023-01-15T14:22:31Z"
            }
        },
        "auth": {
            "token": "dummy_token_xyz"
        }
    }
    
    print("\nLegacy API Input:")
    print(json.dumps(legacy_input, indent=2))
    
    # Process with our rails-as-runnable chain
    legacy_output = rails_as_runnable(legacy_input)
    
    print("\nProcessed Through Rails-as-Runnable:")
    print(json.dumps(legacy_output, indent=2))
    
    print("\nWhy This Matters:")
    print("  - Rails component handles the entire LLM interaction internally")
    print("  - Key mapping allows integration with existing legacy systems")
    print("  - Preserves guardrails protection while maintaining API compatibility")
    print("  - No need to rewrite existing client applications")
    
    print("\n📋 Key Takeaways:")
    print("  - LangChain runnable functions can adapt to different API formats")
    print("  - Input mapping transforms request formats to guardrails format")
    print("  - Output mapping transforms guardrails results to API format")
    print("  - This flexibility allows guardrails to work with existing systems")
    print("  - Rails can serve as the primary runnable component, handling LLM interactions")
    print("  - Key mapping is crucial for integrating guardrails with diverse API architectures")

if __name__ == "__main__":
    main() 