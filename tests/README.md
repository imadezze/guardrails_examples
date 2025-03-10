# Tests for NeMo Guardrails with LangChain Examples

This directory contains tests for the NeMo Guardrails with LangChain integration examples.

## Test Structure

- **Unit Tests**: Test individual components with mocked dependencies
- **Integration Tests**: Test example code end-to-end with mocked external services
- **Common Test Utilities**: Shared test fixtures and mock objects

## Running Tests

You can run the tests using the provided runner script:

```bash
# Run all tests
python run_tests.py

# Run tests for a specific example
python run_tests.py -k add_guardrails_to_chain

# Run tests with coverage
python run_tests.py --cov=examples

# Run tests with verbose output
python run_tests.py -v
```

Alternatively, you can run pytest directly:

```bash
pytest -xvs tests/
```

## Test Files

- `conftest.py` - Common pytest fixtures
- `utils.py` - Test utility functions and mock classes
- `test_add_guardrails_to_chain.py` - Unit tests for the first example
- `test_example_add_guardrails_to_chain.py` - Integration tests for the first example

## Writing New Tests

When adding tests for a new example:

1. Create a file named `test_<example_name>.py` for unit tests
2. Create a file named `test_example_<example_name>.py` for integration tests
3. Use the existing tests as a template
4. Make use of common fixtures from `conftest.py` 
5. Add appropriate mocks to avoid external API calls

## Mocking Strategy

The tests use mocking to avoid making actual API calls:

- LLMs are mocked to return predefined responses
- Toxicity checkers are mocked to simulate safe or unsafe content
- External services (e.g., OpenAI, LangSmith) are mocked

This approach allows the tests to run quickly and reliably without requiring API keys or network connectivity. 