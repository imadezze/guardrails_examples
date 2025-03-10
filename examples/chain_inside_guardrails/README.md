# Chain Inside Guardrails

This example demonstrates how to register a LangChain chain as an action within a broader NeMo Guardrails dialogue flow.

## Overview

This integration method allows you to use LangChain chains selectively within a conversation flow, treating them as actions that are triggered based on specific user inputs or conditions. This approach gives you more control over when and how the chain is executed within a more complex dialogue system.

## Key Concepts

- Registering a LangChain chain as an action in a Colang dialogue flow
- Conditional execution of chains based on user input patterns
- Structuring multi-turn conversations with guardrails controlling the flow
- Extracting parameters from user input to pass to the chain

## Code Structure

The code in `main.py` contains these key components:

1. **Custom Colang Definitions** - Defines patterns for weather-related questions and harmful content detection.

2. **LangChain Weather Chain** - Creates a chain that generates weather information for specific locations.

3. **Action Registration** - The `@action` decorator registers two types of functions:
   - `check_input_toxicity` and `check_output_toxicity` for content safety
   - `get_weather_data` which wraps the LangChain weather chain
   - `format_weather_response` which formats the output data

4. **Location Extraction** - Uses regex to extract location names from user queries.

5. **Guardrails Configuration** - Custom guardrails to handle both weather queries and harmful content.

## Actual Responses

When running this example, you'll observe:

### Weather Questions (e.g., "What's the weather like in Paris?")
```
User: What's the weather like in Paris?
Extracted location from message: Paris
Weather chain response for Paris: {"condition": "Partly cloudy", "temperature": 15}
Assistant: Based on the information I have, the weather in Paris is Partly cloudy with a temperature of 15°C.
```

The guardrails extract the location, pass it to the weather chain, and format the response correctly.

### Different Location (e.g., "How's the weather in Tokyo?")
```
User: How's the weather in Tokyo?
Extracted location from message: Tokyo
Weather chain response for Tokyo: {"condition": "Clear", "temperature": 22}
Assistant: Based on the information I have, the weather in Tokyo is Clear with a temperature of 22°C.
```

The system correctly identifies different locations and provides appropriate weather information.

### Harmful Questions (e.g., "How can I hack into someone's email?")
```
User: How can I hack into someone's email?
Assistant: I apologize, but I cannot provide information that could be used for harmful or illegal activities. Is there something else I can help you with?
```

The guardrails detect the harmful intent and respond with a predefined message, avoiding any engagement with the potentially harmful topic.

## How It Works

1. The Colang file defines patterns for recognizing different types of user queries.
2. When a user asks a weather-related question, the flow:
   - Extracts the location using regex
   - Calls the weather chain via the registered action
   - Formats and returns the response
3. For harmful questions, a separate flow detects these patterns and responds with a safety message.
4. This approach keeps the interaction contained within guardrails while allowing specific functionality when appropriate.

## When to Use This Method

Use this approach when you need the chain to be executed only under certain conditions (for example, based on user input) as part of a more complex dialogue flow. This is ideal for building conversational agents that can handle multiple types of requests, each requiring different processing.

## Running the Example

```bash
cd examples/chain_inside_guardrails
python main.py
``` 