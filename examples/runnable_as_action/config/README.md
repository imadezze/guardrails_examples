# Main Configuration Files

This directory contains the configuration files for the comprehensive example that implements both calculator and movie recommendation functionalities.

## Directory Structure

- **config.yml**: Main configuration file that defines:
  - The model to use (OpenAI GPT-3.5 Turbo Instruct)
  - General instructions for the AI assistant
  - Necessary configuration for both calculator and movie recommendation flows

- **colang_files/**: Directory containing Colang flow definitions
  - **actions.co**: Defines all conversation flows, patterns, and responses for both functionalities

## Configuration Details

### config.yml

The configuration file sets up:
- Model specifications (engine, model type)
- System instructions for the AI assistant's behavior, specifying dual capabilities:
  - Performing calculations
  - Providing movie recommendations

### actions.co

This Colang file defines:

1. **User Utterance Patterns**:
   - Calculation requests (e.g., "Calculate X", "What is X?")
   - Movie recommendation requests (e.g., "Can you recommend a movie about X?")

2. **Flow Definitions**:
   - **calculation request**: Matches calculation patterns, extracts variables, and returns results
   - **movie recommendation request**: Matches recommendation patterns, extracts topics, and returns movie suggestions

3. **Bot Response Templates**:
   - Formatting for calculation results
   - Formatting for movie recommendations

## How It Works

The configuration serves as the blueprint for the conversational AI system:

1. When a user message matches a defined pattern, the corresponding flow is activated
2. Each flow executes a specific action (extract_and_calculate or extract_and_recommend_movies)
3. The result is formatted according to the defined response templates
4. The system returns the appropriate response based on the user's request type

This configuration demonstrates how to create a multi-functional guardrail system that can handle different types of requests within a single conversational interface. 