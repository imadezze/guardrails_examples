# LangSmith Integration

This example demonstrates how to use LangSmith for tracing, monitoring, and debugging LangChain components that have been wrapped with NeMo Guardrails.

## Overview

LangSmith integration provides observability into your guardrailed chains, allowing you to trace the execution path, understand how inputs are transformed through the guardrails, and identify potential issues. This approach is particularly valuable during development and troubleshooting of complex guardrailed systems.

## Key Concepts

- Setting up LangSmith tracing with NeMo Guardrails
- Inspecting how guardrails modify or block inputs/outputs
- Analyzing trace data to understand rail behavior
- Capturing and debugging interactions that trigger safety measures

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

After running the example, you can view the traces in your LangSmith dashboard. 