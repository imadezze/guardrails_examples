# Runnable-As-Action

This example demonstrates how to register LangChain Runnable components as actions within a NeMo Guardrails dialogue flow, allowing conditional execution based on user inputs.

## Overview

The Runnable-As-Action approach allows you to integrate LangChain components as discrete actions within a conversation flow, creating a multi-functional AI assistant that can perform different tasks depending on user requests. This method provides fine-grained control over when each component is executed and how its results are used.

## Key Concepts

- Registering multiple LangChain chains as separate actions
- Using Colang to define conversation flows with action execution
- Robust pattern matching and variable extraction using regex
- Parameter extraction from natural language
- Conditional execution based on conversation context
- Structuring responses based on action outputs
- Debug mode for detailed logging and troubleshooting

## Code Structure

This repository contains three key Python files that demonstrate different aspects of the Runnable-As-Action approach:

1. **main.py** - A comprehensive example demonstrating:
   - Multiple action registrations (calculator and movie recommender)
   - Regex-based pattern extraction for robust variable handling
   - Conditional debugging with detailed logging
   - Two separate conversation flows in a single application
   - Error handling and input validation
   - Clean output formatting

2. **simplified_main.py** - A streamlined calculator-only example focusing on:
   - Basic action registration pattern
   - Essential variable extraction techniques
   - Minimal implementation for calculator functionality
   - Debugging and error handling fundamentals
   - Ideal for beginners to understand core concepts

3. **direct_test.py** - A direct testing utility that:
   - Bypasses the guardrails conversation flow
   - Directly invokes LangChain chains for testing
   - Provides baseline performance metrics
   - Useful for isolating and debugging chain behavior

## Implementation Details

### Key Components

- **Custom Actions**: Functions that wrap LangChain chains, registered with the guardrails system
- **Variable Extraction**: Regex-based extraction to reliably parse user inputs
- **Colang Flows**: Structured conversation patterns defining when actions are triggered
- **Debug Mode**: Comprehensive logging system toggled via environment variables
- **Output Formatting**: Clean presentation of calculation results and recommendations

### How It Works

1. **Request Processing**: User messages are matched against defined patterns in Colang
2. **Variable Extraction**: When a match is found, regex extraction pulls key variables from the message
3. **Action Execution**: The extracted variables are passed to the appropriate LangChain chain
4. **Response Formatting**: Results are formatted according to predefined templates
5. **Error Handling**: Robust error detection and user-friendly error messages

This approach connects natural language understanding (via Colang) with specialized task execution (via LangChain) in a maintainable and extensible way.

## Direct Testing

The `direct_test.py` script provides a way to test the LangChain components directly, without the NeMo Guardrails wrapper. This is useful for:

1. Comparing performance with and without guardrails
2. Debugging issues with the underlying chains
3. Testing prompt templates in isolation
4. Understanding the baseline behavior of the models

### How Direct Testing Works

The direct testing script:
1. Creates the same LangChain components (calculator and movie recommendation chains) used in the main examples
2. Invokes them directly with test inputs
3. Displays the raw results without any guardrails processing

### Direct Testing Components

- **Calculator Chain**: A LangChain composed of a prompt template and the GPT-4o-mini model
- **Movie Recommendation Chain**: A LangChain using GPT-3.5-turbo with a specialized prompt for movie recommendations
- **Direct Invocation**: Calls the chains with test inputs, bypassing the guardrails system
- **Result Formatting**: Simple output that shows the raw responses from the models

### When to Use Direct Testing

Use direct testing when:
- You suspect an issue might be in the guardrails configuration rather than the chains
- You want to see how the raw LLM responses differ from the guardrailed ones
- You're developing new prompts and want to test them quickly
- You need to isolate performance issues between the chains and the guardrails

### Direct Testing Output Example

```
--- Testing Calculator Chain Directly ---

Calculating: 23 + 45
Result: 68

Calculating: sqrt(144) + 10
Result: 22

Calculating: (35 * 12) / 7
Result: 60

--- Testing Movie Recommendation Chain Directly ---

Recommending movies about: space exploration
Recommendations:
1. Interstellar - A group of astronauts travel through a wormhole in search of a new habitable planet for humanity.
2. The Martian - An astronaut becomes stranded on Mars and must use his ingenuity to survive.
3. Apollo 13 - Based on a true story, the film follows NASA's aborted mission to the moon and the efforts to bring the astronauts home safely.
```

This output can be compared with the guardrailed version to understand how the NeMo Guardrails system affects the final responses.

## Benefits of Guardrails

The examples in this repository demonstrate several key benefits of using NeMo Guardrails to wrap LangChain components as actions rather than using them directly:

### 1. Enhanced Safety and Control

Without guardrails, LLM chains can sometimes produce unexpected or problematic outputs if they receive unusual inputs. For example:

```
# Direct LLM Chain (without guardrails)
Input: Calculate 0/0
Result: Error: Division by zero is undefined.

# With Guardrails
User: Calculate 0/0
Assistant: I'm sorry, I couldn't calculate '0/0'. Please check if the expression is valid.
```

The guardrails system provides proper error handling and user-friendly messages.

### 2. Contextual Understanding

Guardrails can help the system understand which action to invoke based on natural language context:

```
# Direct Chain (requires exact parameter matching)
calculator_chain.invoke({"expression": "23 + 45"})  # Works
calculator_chain.invoke({"expression": "what is 23 + 45"})  # Might fail

# With Guardrails
User: What is 23 + 45?  
User: Calculate 23 + 45.
User: Could you compute 23 + 45 for me?
```

All these variations work with guardrails because the system parses intention from natural language.

### 3. Multi-functionality in One Interface

Without guardrails, you would need separate interfaces for each functionality:

```
# Without Guardrails
calculator_chain.invoke({"expression": "23 + 45"})
movie_chain.invoke({"topic": "space exploration"})

# With Guardrails
User: Calculate 23 + 45
User: Can you recommend a movie about space exploration?
```

Guardrails provide a unified conversation interface that intelligently routes to the appropriate action.

### 4. Improved Error Handling

Guardrails provide robust handling of edge cases:

```
# Without Guardrails (direct chain)
Input: Calculate the meaning of life
Result: [Unpredictable response or error]

# With Guardrails
User: Calculate the meaning of life
Assistant: I couldn't understand the expression. Please provide it in the format 'Calculate X' where X is a mathematical expression.
```

The guardrails extract patterns correctly and provide helpful guidance when inputs don't match expected formats.

### 5. Debugging Capabilities

With the DEBUG mode, guardrails provide detailed insights into the system's decision-making process:

```
DEBUG=true python main.py

Output includes:
- Which patterns were matched
- How variables were extracted
- Which actions were triggered
- Rails that were activated
- Complete processing flow
```

This makes it much easier to diagnose and fix issues compared to directly using LangChain components.

## Actual Responses

When running this example, you'll observe these types of responses:

### Calculator Functionality
```
User: Calculate 23 + 45
Assistant: The result of the calculation is 68

User: Calculate sqrt(144) + 10
Assistant: The result of the calculation is 22

User: Calculate (35 * 12) / 7
Assistant: The result of the calculation is 60
```

The calculator functionality demonstrates how mathematical expressions are extracted, processed, and returned with clear formatting.

### Movie Recommendation Functionality
```
User: Can you recommend a movie about space exploration?
Assistant: Here are some movie recommendations:
1. Interstellar - A group of astronauts travel through a wormhole near Saturn in search of a new habitable planet for humanity.
2. The Martian - After being stranded on Mars, an astronaut must use his wit and ingenuity to survive until rescue.
3. Apollo 13 - Based on a true story, this film follows the failed Apollo 13 mission and the efforts to bring the astronauts safely back to Earth.

User: What's a good film about artificial intelligence?
Assistant: Here are some movie recommendations:
1. Ex Machina - A young programmer is selected to participate in a groundbreaking experiment in synthetic intelligence by evaluating the human qualities of a highly advanced humanoid AI.
2. Her - A lonely writer develops an unlikely relationship with an operating system designed to meet his every need.
3. I, Robot - In a future where robots serve humanity, a detective investigates a crime that may have been committed by a robot.
```

The movie recommendation functionality demonstrates how topic extraction, specialized chain invocation, and formatted responses work together to provide a natural conversation experience.

## When to Use This Method

Use this approach when you want to build a versatile assistant that can:

- Handle multiple distinct functions through a unified conversation interface
- Conditionally execute chains or external tools as one action among many
- Enable flexible branching and conditional dialogues
- Maintain control over which components are activated in response to specific user requests
- Integrate specialized tools for different tasks (calculations, recommendations, etc.)

## Running the Examples

### Main Example (Calculator and Movie Recommendations)

```bash
cd examples/runnable_as_action
python main.py
```

### Simplified Example (Calculator Only)

```bash
cd examples/runnable_as_action
python simplified_main.py
```

### Direct Testing (Without Guardrails)

```bash
cd examples/runnable_as_action
python direct_test.py
```

### Debug Mode

To run any example with detailed debug information:

```bash
DEBUG=true python main.py
# or
DEBUG=true python simplified_main.py
```

The debug mode provides comprehensive insights into:
- Variable extraction details
- Action execution steps
- Input and output processing
- Activated rails and decisions
- Error detection and handling

This example demonstrates how different LangChain chains can be conditionally executed based on user requests, with robust error handling and variable extraction to ensure reliable operation.