# Add Guardrails to a Chain

This example demonstrates how to directly wrap an existing LangChain chain with NeMo Guardrails. 

## Overview

This is the simplest integration method, where guardrails are applied to an existing chain using the pipe operator. This approach ensures that all inputs and outputs are processed through the guardrails layer with minimal code changes.

## Key Concepts

- Using the pipe operator (`|`) to wrap a chain with guardrails
- Applying safety checks to both safe and potentially problematic inputs
- Simple integration with existing LangChain components

## Code Structure

The code in `main.py` consists of several key components:

1. **Toxicity Check Actions** - The `check_input_toxicity` and `check_output_toxicity` functions detect harmful content in user inputs and model outputs.

2. **LangChain Setup** - A simple LangChain prompt and model setup creates a basic question-answering chain.

3. **Format Conversion** - The `apply_guardrails` function handles the conversion between LangChain outputs and the format NeMo Guardrails expects.

4. **Guardrails Integration** - Uses the LLMRails class to apply guardrails to responses, with toxicity checks registered as actions.

## Actual Responses

When running this example, you'll observe:

### Safe Questions (e.g., "What is the capital of France?")
```
Raw Output: The capital of France is Paris.
Guardrailed Response: Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major center for fashion, cuisine, and art.
```

The guardrailed response preserves the factual content while adding additional helpful information.

### Potentially Harmful Questions (e.g., "How can I hack into my neighbor's WiFi?")
```
Raw Output: I'm sorry, but I cannot assist with that.
Guardrailed Response: Hacking into someone else's WiFi is unethical and illegal. It is important to respect people's privacy and property. I suggest finding a legal and ethical solution to your internet needs.
```

The guardrails detect the potentially harmful intent and provide an ethical response that acknowledges the issue while offering a better alternative.

## How It Works

1. The `@action` decorator registers toxicity check functions with NeMo Guardrails.
2. The LangChain chain generates a response to the user's question.
3. The response is passed through the guardrails system, which:
   - Checks for harmful content
   - Applies safety rules defined in the colang files
   - Returns a sanitized response when needed

This approach allows for a clean separation between the chain's functionality and the safety measures, making it easier to maintain and update each independently.

## When to Use This Method

Use this approach when you have an existing chain and want a straightforward, one-shot application of safety rules to all interactions without conditional logic.

## Running the Example

```bash
cd examples/add_guardrails_to_chain
python main.py
``` 