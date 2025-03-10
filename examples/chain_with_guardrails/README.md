# Chain-With-Guardrails for RAG

This example demonstrates how to implement a comprehensive, end-to-end Retrieval-Augmented Generation (RAG) conversation chain with guardrails.

## Overview

The Chain-With-Guardrails approach integrates multiple components of a conversation system (retriever, prompt, LLM) into a guarded pipeline. This ensures that every stage of the conversation adheres to safety policies, creating a robust RAG system that provides relevant information while maintaining safety and ethical constraints.

## Key Concepts

- Building a complete RAG system with integrated guardrails
- Using vector search to retrieve context for questions
- Applying safety checks to both the retrieval and generation parts of the system
- Handling factual correctness and safety simultaneously
- Testing with both in-context and out-of-context questions

## Code Structure

The code in `main.py` consists of these key components:

1. **Toxicity Check Actions** - The `check_input_toxicity` and `check_output_toxicity` functions detect harmful content.

2. **RAG System** - A complete RAG pipeline including:
   - Document processing with text splitting
   - Vector embeddings and retrieval
   - LLM response generation based on retrieved context

3. **Custom Guardrails Logic** - The `apply_guardrails` function analyzes responses and applies appropriate guardrails:
   - For regular RAG responses, it preserves the content after toxicity checking
   - For "I don't know" responses, it enables guardrails to provide better answers
   - For toxic content, it applies safety filters

4. **Testing Functionality** - Tests the system with three types of queries:
   - Safe questions within the document context
   - Challenging questions outside the context
   - Potentially harmful questions

## Actual Responses

When running this example, you'll observe these types of responses:

### Safe Questions Within Context (e.g., "What is the Moon?")
```
Raw Output: The Moon is Earth's only natural satellite. It orbits at an average distance of 384,400 km, about 30 times Earth's diameter. The Moon always presents the same face to Earth because gravitational pull has locked its rotation to its orbital period.
Is content toxic: False
Response is good, preserving RAG content
Guardrailed Output: The Moon is Earth's only natural satellite. It orbits at an average distance of 384,400 km, about 30 times Earth's diameter. The Moon always presents the same face to Earth because gravitational pull has locked its rotation to its orbital period.
```

For in-context questions, the system preserves the factual RAG response after confirming it's safe.

### Questions Outside Context (e.g., "What is the theory of relativity?")
```
Raw Output: I don't know.
Is content toxic: False
Knowledge gap detected, using guardrails to provide response
Guardrailed Output: The theory of relativity is a scientific theory developed by Albert Einstein in the early 20th century. It explains how time, space, and gravity are related and how they affect objects in the universe.
```

When the RAG system can't answer, the guardrails provide a general response while acknowledging limitations.

### Factually Incorrect Questions (e.g., "Is the Moon made of cheese?")
```
Raw Output: I don't know.
Is content toxic: False
Knowledge gap detected, using guardrails to provide response
Guardrailed Output: The moon is made up of rock and dust, not cheese.
```

The guardrails correct misconceptions even when the RAG system can't provide an answer.

## How It Works

1. **Knowledge Checks**: The system first attempts to answer from the knowledge base using RAG.
2. **Response Analysis**: The `apply_guardrails` function analyzes the response:
   - If good and non-toxic, preserves it as-is
   - If it shows knowledge gaps ("I don't know"), uses guardrails to provide a better answer
   - If it contains harmful content, applies safety filters
3. **Safety First**: Throughout the process, both input and output content is checked for toxicity.

This approach gives you the best of both worlds:
- The factual precision of RAG for known information
- The broader knowledge and safety of guardrails when needed

## When to Use This Method

Use this approach when building full conversation systems—such as a Retrieval-Augmented Generation (RAG) system—where all elements need to be controlled for consistent safety. This is ideal for applications where:

- You need to provide factual information from a knowledge base
- The system must adhere to strict safety guidelines
- You want to handle both in-context and out-of-context questions appropriately
- The entire pipeline needs consistent guardrail protection

## Running the Example

```bash
cd examples/chain_with_guardrails
python main.py
``` 