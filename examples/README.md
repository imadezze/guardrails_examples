# NeMo Guardrails with LangChain Integration Examples

This directory contains examples demonstrating different methods for integrating NeMo Guardrails with LangChain.

## Examples Overview

| Example | Description | When to Use |
|---------|-------------|-------------|
| [add_guardrails_to_chain](./add_guardrails_to_chain/) | Directly wraps an existing LangChain chain with guardrails | When you need a simple, one-shot application of safety rules |
| [chain_inside_guardrails](./chain_inside_guardrails/) | Uses a LangChain chain as an action within a guardrails flow | When chains should be executed conditionally in a conversation |
| [langsmith_integration](./langsmith_integration/) | Adds LangSmith tracing to guardrailed chains | For debugging and monitoring guardrail behavior |
| [runnable_rails](./runnable_rails/) | Demonstrates advanced RunnableRails features | When you need fine-grained control over inputs/outputs |
| [chain_with_guardrails](./chain_with_guardrails/) | Implements a comprehensive RAG system with guardrails | For complete retrieval-augmented generation systems |
| [runnable_as_action](./runnable_as_action/) | Registers chains as actions to use in conversation flows | For building multi-purpose assistants with specialized tools |

## Comparison of Methods

### Integration Level

- **Low-level integration**: `runnable_rails` gives you the most control over how guardrails are applied
- **Mid-level integration**: `add_guardrails_to_chain` and `chain_with_guardrails` provide ready-to-use patterns
- **High-level integration**: `chain_inside_guardrails` and `runnable_as_action` embed chains within conversation flows

### Control Flow

- **Always active**: `add_guardrails_to_chain` applies guardrails to every input/output
- **Conditional execution**: `chain_inside_guardrails` and `runnable_as_action` execute chains only when needed
- **Pipeline integration**: `chain_with_guardrails` integrates guardrails into a multi-step process

### Use Case Fit

- **Simple Q&A**: `add_guardrails_to_chain` is sufficient for straightforward question-answering
- **RAG systems**: `chain_with_guardrails` is designed for retrieval-augmented generation
- **Multi-functional assistants**: `runnable_as_action` works best for assistants with multiple capabilities
- **Weather, calculations, etc.**: `chain_inside_guardrails` is ideal for domain-specific functionality

### Development Stage

- **Prototyping**: Start with `add_guardrails_to_chain` for quick implementation
- **Development**: Use `langsmith_integration` to understand guardrail behavior
- **Advanced development**: Implement `runnable_rails` for custom logic
- **Production**: Consider `chain_with_guardrails` or `runnable_as_action` for robust systems

## Common Resources

The [common](./common/) directory contains shared resources used across examples:

- Configuration files
- Utility functions
- Shared data

## Running Examples

Each example can be run independently. Navigate to the example directory and run:

```bash
python main.py
```

Or run all examples with:

```bash
python ../run_all_examples.py
``` 