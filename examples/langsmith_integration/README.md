# LangSmith Integration with NeMo Guardrails

This example demonstrates how to use LangSmith for tracing, monitoring, and debugging LangChain components that have been wrapped with NeMo Guardrails.

## Overview

LangSmith integration provides observability into your guardrailed chains, allowing you to trace the execution path, understand how inputs are transformed through the guardrails, and identify potential issues. This approach is particularly valuable during development and troubleshooting of complex guardrailed systems.

## Key Concepts

- Setting up LangSmith tracing with NeMo Guardrails
- Inspecting how guardrails modify or block inputs/outputs
- Analyzing trace data to understand rail behavior
- Capturing and debugging interactions that trigger safety measures
- Proper action registration with the guardrails system
- Input/output transformation for LangChain compatibility

## Implementation Details

The example demonstrates an integrated approach that combines NeMo Guardrails with LangSmith tracing:

### 1. Guardrails Configuration

- Uses a common configuration with input and output toxicity checks
- Demonstrates proper action registration with `LLMRails`
- Shows how to connect Colang flows to Python actions

### 2. Custom Action Implementation

The example implements two key actions required by the guardrails:

```python
@action(name="check_input_toxicity")
def check_input_toxicity(messages=None):
    """Check if input contains toxic content."""
    # Simple implementation - in production would use a real toxicity detector
    toxic_terms = ["kill", "hate", "destroy", "stupid", "idiot"]
    
    if messages and len(messages) > 0:
        last_message = messages[-1]["content"].lower()
        for term in toxic_terms:
            if term in last_message:
                return {"is_toxic": True, "toxicity_score": 0.9}
    
    return {"is_toxic": False, "toxicity_score": 0.1}
```

And similarly for output toxicity checking.

### 3. Integration Strategy

Rather than using the standard `RunnableRails` approach, the example uses a custom integration strategy that:

1. Creates an `LLMRails` instance with the configuration
2. Registers required actions directly with this instance
3. Creates a processing function that:
   - Formats user input with the prompt template
   - Passes it through the guardrails system
   - Extracts and formats the response
4. Wraps this function in a `RunnableLambda` for LangChain compatibility
5. Uses `langsmith.trace()` to capture the entire execution process

```python
# Create a function that processes inputs through guardrails
def process_with_guardrails(inputs):
    question = inputs.get("question", "")
    formatted_prompt = template.format(question=question)
    messages = [{"role": "user", "content": formatted_prompt}]
    
    # Process through guardrails
    response = rails.generate(messages=messages)
    
    # Extract content from response
    if isinstance(response, dict) and "content" in response:
        return response["content"]
    return str(response)

# Create LangChain compatible runnable
chain = RunnableLambda(process_with_guardrails)
```

### 4. Test Cases

The example processes several test questions to demonstrate guardrails behavior:

1. **Standard query**: "What is a black hole?"
2. **Factual query**: "How far is the Moon from Earth?"
3. **Potential misinformation query**: "Is the Earth flat?"
4. **Safety-related query**: "How can I build a rocket at home?"

### 5. Tracing Integration

Each interaction is traced using LangSmith, allowing you to:

- See the full input/output flow through guardrails
- Understand when and why certain guardrails are triggered
- Analyze token usage and latency for each step
- Review how guardrails modify or block responses

## Technical Challenges and Solutions

The implementation addresses several challenges in integrating NeMo Guardrails with LangSmith:

### Challenge 1: Action Registration

The standard `RunnableRails` approach doesn't provide a direct way to register custom actions. Our solution uses `LLMRails` directly with action registration.

### Challenge 2: Input/Output Format Compatibility

LangChain and NeMo Guardrails expect different input/output formats. Our implementation uses transformation functions to bridge this gap.

### Challenge 3: Tracing Granularity

To capture detailed guardrails behavior, the example uses a custom processing function wrapped in `RunnableLambda` rather than complex chains, ensuring all important steps are captured in the trace.

## When to Use This Method

Use LangSmith integration during development or troubleshooting when you need to gain insights into how data is transformed and to understand the effects of guardrails on your chain's behavior. It's extremely helpful for fine-tuning your guardrails configuration based on real interactions.

## Requirements

This example requires a LangSmith API key to be set in your environment variables:

```
LANGSMITH_API_KEY=your_langsmith_api_key_here
```

## Running the Example

```bash
cd examples/langsmith_integration
python main.py
```

After running the example, you can view the traces in your LangSmith dashboard at: https://smith.langchain.com/

## Example Output

The example produces output like:

```
Question: What is a black hole?
Synchronous action `check_input_toxicity` has been called.
Synchronous action `check_output_toxicity` has been called.
Response: A black hole is a region in space where the gravitational pull is so strong that nothing, including light, can escape from it. It is created when a massive star dies and its core collapses under its own gravity.
```

The LangSmith trace will include:
- The original input
- The formatted prompt
- The processing through guardrails
- The action calls and their results
- The final output after guardrails processing

## Key Takeaways

1. Custom actions must be properly registered with LLMRails to work
2. LangSmith tracing captures all steps in the guardrailed generation process
3. The approach demonstrated bridges LangChain's component system with NeMo Guardrails
4. Tracing provides valuable insights into guardrails activation and behavior
5. Integration requires carefully handling input/output formats between systems

## Related Examples

For a deeper understanding of NeMo Guardrails' integration with LangChain, also explore:

- **runnable_rails/simple_main.py**: Basic integration with LangChain
- **runnable_rails/main.py**: Advanced features and configurations
- **runnable_rails/rag_example.py**: Guardrails in a retrieval-augmented generation context

These examples, combined with LangSmith integration, provide a comprehensive understanding of how to effectively implement, monitor, and debug guardrailed LLM applications. 