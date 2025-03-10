# Simplified Calculator Configuration

This directory contains the configuration files for the simplified calculator example.

## Directory Structure

- **config.yml**: Main configuration file that defines:
  - The model to use (OpenAI GPT-3.5 Turbo Instruct)
  - General instructions for the AI assistant
  - Rails configuration specifying the "calculation request" flow

- **colang_files/**: Directory containing Colang flow definitions
  - **calculator.co**: Defines the calculator conversation flow patterns and responses

## Configuration Details

### config.yml

The configuration file sets up:
- Model specifications (engine, model type)
- System instructions for the AI assistant's behavior
- Dialog rails for the calculation flow

### calculator.co

This Colang file defines:
- User utterance patterns for calculation requests
- The calculation request flow which:
  1. Identifies when a user asks for a calculation
  2. Executes the extract_and_calculate action with the user's message
  3. Provides the calculation result back to the user
- Bot response templates for formatting the calculation result

## How It Works

When a user message matches one of the defined patterns for calculations (e.g., "Calculate X", "What is X?", "Compute X"), the guardrails system:

1. Activates the "calculation request" flow
2. Passes the user message to the extract_and_calculate function
3. The function extracts the mathematical expression using regex
4. The expression is calculated using a LangChain calculator
5. The result is formatted and returned to the user

This configuration demonstrates how to create a focused, single-purpose guardrail for mathematical calculations with proper error handling and response formatting. 