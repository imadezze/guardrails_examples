# NeMo Guardrails with LangChain Examples

This repository contains examples for integrating NeMo Guardrails with LangChain using various methods.

## Methods Covered

1. **Add Guardrails to a Chain** - Directly wraps an existing LangChain chain
2. **Using a Chain inside Guardrails** - Registers a LangChain chain as an action within a guardrails flow
3. **LangSmith Integration** - Integrates LangSmith for monitoring and debugging
4. **RunnableRails** - Uses the core interface to wrap LangChain components with Guardrails
5. **Chain-With-Guardrails** - Implements a comprehensive conversation chain
6. **Runnable-As-Action** - Registers a LangChain chain as an action to be invoked within a flow

## Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd nemoguardrails
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit the .env file with your API keys
```

## Running the Examples

Each example is contained in its own directory. To run an example:

```bash
cd examples/<example_directory>
python main.py
```

## Running All Examples

To run all examples at once:

```bash
python run_all_examples.py
```

## Running Tests

The repository includes a comprehensive test suite to verify the examples work correctly:

```bash
# Run all tests
python run_tests.py

# Run tests for a specific example
python run_tests.py -k add_guardrails_to_chain

# Run tests with coverage
python run_tests.py --cov=examples
```

See the `tests` directory for more details on the testing approach.

## Documentation

For more detailed information, refer to:
- [NeMo Guardrails Documentation](https://docs.nvidia.com/nemo/guardrails/latest/user-guides/langchain/index.html)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)

## License

This project is provided for educational purposes only. 