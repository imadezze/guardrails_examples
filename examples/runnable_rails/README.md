# RunnableRails with Advanced Features

This example demonstrates how to use RunnableRails with advanced features like key mapping, prompt passthrough, and streaming support.

## Overview

RunnableRails is the core interface used to wrap LangChain components with NeMo Guardrails. This example showcases its advanced capabilities, giving you fine-grained control over how inputs and outputs are processed when using guardrails with LangChain components.

## Key Concepts

- Basic RunnableRails with default configuration
- Custom key mapping for inputs and outputs
- Streaming support for incremental response generation
- Prompt passthrough to preserve original inputs
- Handling multi-input prompts with guardrails

## When to Use This Method

Use RunnableRails with advanced features when your application requires fine-tuned control over how messages are handled through the guardrails process. This approach is particularly useful when:

- You need to map between different input/output formats
- You want to preserve original inputs for later use
- Your application requires streaming responses
- You're working with complex chains that have multiple inputs/outputs

## Running the Example

```bash
cd examples/runnable_rails
python main.py
```

The example includes four different ways to use RunnableRails, each demonstrating a specific advanced feature. 