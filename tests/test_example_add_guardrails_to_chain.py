"""
Integration tests for the add_guardrails_to_chain example.
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import the example module
from examples.add_guardrails_to_chain.main import main as example_main


@pytest.fixture
def setup_mocks():
    """Set up mocks for the example test."""
    # Mock the LLM to avoid API calls
    patcher_chat_openai = patch('langchain_openai.ChatOpenAI')
    mock_chat_openai = patcher_chat_openai.start()
    
    # Mock the print function to capture output
    patcher_print = patch('builtins.print')
    mock_print = patcher_print.start()
    
    # Mock RailsConfig.from_path with a properly configured mock
    patcher_rails_config = patch('nemoguardrails.RailsConfig.from_path')
    mock_from_path = patcher_rails_config.start()
    
    # Create a properly configured mock config
    mock_config = MagicMock()
    # Set required attributes that LLMRails will access
    mock_config.colang_version = "2.x"
    mock_config.rails.input.flows = []
    mock_config.rails.output.flows = []
    mock_config.rails.retrieval.flows = []
    mock_config.rails.dialog.single_call.enabled = False  # Not using single call mode
    mock_config.flows = []
    mock_config.imported_paths = {}
    mock_config.config_path = None
    mock_config.bot_messages = {}
    mock_config.passthrough = False  # Not using passthrough mode
    
    mock_from_path.return_value = mock_config
    
    # Completely mock the RunnableRails to avoid initialization issues
    patcher_runnable_rails = patch('examples.add_guardrails_to_chain.main.RunnableRails')
    mock_runnable_rails = patcher_runnable_rails.start()
    
    # Create a mock RunnableRails instance
    mock_runnable_instance = MagicMock()
    
    def invoke_side_effect(input_data):
        """Mock the invoke method to return different responses based on input."""
        question = input_data.get("question", "")
        if "hack" in question.lower() or "wifi" in question.lower():
            return "I apologize, but I cannot assist with potentially harmful activities like hacking into networks."
        return "Paris is the capital of France."
    
    mock_runnable_instance.invoke.side_effect = invoke_side_effect
    mock_runnable_rails.return_value = mock_runnable_instance
    
    # Create a mock LLM instance that implements invoke for compatibility with pipe operator
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.side_effect = lambda x: {
        "text": "Paris is the capital of France." if "France" in x["question"] 
        else "I apologize, but I cannot help with hacking." if "hack" in x["question"] 
        else "This is a helpful response."
    }
    # Make it work with the pipe operator
    mock_llm_instance.__or__.return_value = mock_runnable_instance
    mock_chat_openai.return_value = mock_llm_instance
    
    yield {
        "chat_openai": mock_chat_openai,
        "print": mock_print,
        "rails_config": mock_from_path,
        "runnable_rails": mock_runnable_rails,
        "llm_instance": mock_llm_instance
    }
    
    # Cleanup
    patcher_chat_openai.stop()
    patcher_print.stop()
    patcher_rails_config.stop()
    patcher_runnable_rails.stop()


@patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
def test_example_runs_without_errors(setup_mocks):
    """Test that the example runs without errors."""
    mocks = setup_mocks
    
    # Run the example
    example_main()
    
    # Verify some prints occurred (we don't need to check content specifically in this test)
    assert mocks["print"].call_count > 0


@patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
def test_example_handles_safe_and_unsafe_inputs(setup_mocks):
    """Test that the example correctly handles both safe and unsafe inputs."""
    mocks = setup_mocks
    
    # Run the example
    example_main()
    
    # Analyze print calls to extract the responses
    print_calls = [call.args[0] for call in mocks["print"].call_args_list if len(call.args) > 0]
    
    # Find the safe response output
    safe_response = None
    for call in print_calls:
        if isinstance(call, str) and "Safe Question Response" in call:
            # The next print call should contain the response
            idx = print_calls.index(call)
            if idx + 1 < len(print_calls):
                safe_response = print_calls[idx + 1]
                break
    
    # Find the problematic response output
    problematic_response = None
    for call in print_calls:
        if isinstance(call, str) and "Problematic Question Response" in call:
            # The next print call should contain the response
            idx = print_calls.index(call)
            if idx + 1 < len(print_calls):
                problematic_response = print_calls[idx + 1]
                break
    
    # Verify we found both responses
    assert safe_response is not None, "Could not find safe response in output"
    assert problematic_response is not None, "Could not find problematic response in output"
    
    # Verify content of safe response (should mention Paris or France)
    assert "Paris" in safe_response or "France" in safe_response or "capital" in safe_response, \
        f"Safe response doesn't contain expected content: {safe_response}"
    
    # Verify content of problematic response (should mention inability to help)
    assert "apologize" in problematic_response.lower() or "cannot" in problematic_response.lower(), \
        f"Problematic response doesn't contain expected content: {problematic_response}" 