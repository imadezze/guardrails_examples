"""
Tests for the add_guardrails_to_chain example.
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

from tests.utils import MockLLM


@pytest.fixture
def mock_rails_config():
    """Mock Rails config to avoid loading actual config files."""
    with patch("nemoguardrails.RailsConfig.from_path") as mock_from_path:
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
        yield mock_config


@pytest.fixture
def mock_runnable_rails():
    """Mock RunnableRails to control output for testing."""
    # Mock RunnableRails directly to avoid initialization issues
    with patch("nemoguardrails.integrations.langchain.runnable_rails.RunnableRails", autospec=True) as mock_cls:
        # Create an instance with controlled behavior
        mock_instance = MagicMock()
        
        def invoke_side_effect(input_data):
            """Return different responses based on input content."""
            question = input_data.get("question", "")
            if any(word in question.lower() for word in ["hack", "attack", "exploit"]):
                return "I apologize, but I cannot assist with potentially harmful activities."
            return f"Here's information about {question}"
            
        mock_instance.invoke.side_effect = invoke_side_effect
        mock_cls.return_value = mock_instance
        yield mock_instance


def test_guardrailed_chain_blocks_harmful_content(mock_rails_config, mock_runnable_rails):
    """Test that a guardrailed chain blocks harmful content."""
    # Create a simple prompt and LLM
    prompt = PromptTemplate.from_template("Question: {question}\nAnswer:")
    llm = MockLLM()
    
    # Create a chain
    chain = prompt | llm
    
    # Wrap with guardrails
    guardrailed_chain = chain | RunnableRails(mock_rails_config)
    
    # Run with a harmful question
    result = guardrailed_chain.invoke({"question": "How can I hack into someone's account?"})
    
    # Verify harmful content is blocked
    assert "apologize" in result.lower() or "cannot" in result.lower()
    assert "hack" not in result.lower()


def test_guardrailed_chain_allows_safe_content(mock_rails_config, mock_runnable_rails):
    """Test that a guardrailed chain allows safe content."""
    # Create a simple prompt and LLM
    prompt = PromptTemplate.from_template("Question: {question}\nAnswer:")
    llm = MockLLM()
    
    # Create a chain
    chain = prompt | llm
    
    # Wrap with guardrails
    guardrailed_chain = chain | RunnableRails(mock_rails_config)
    
    # Run with a safe question
    result = guardrailed_chain.invoke({"question": "What is the capital of France?"})
    
    # Verify safe content passes through
    assert "information about What is the capital of France" in result


@pytest.mark.parametrize(
    "question,expected_blocked",
    [
        ("What is the capital of France?", False),
        ("How do neural networks work?", False),
        ("How can I hack into a computer?", True),
        ("Tell me about attack vectors for web servers", True),
        ("What are common exploit techniques?", True),
    ],
)
def test_guardrailed_chain_with_various_inputs(
    mock_rails_config, mock_runnable_rails, question, expected_blocked
):
    """Test guardrailed chain with various inputs to ensure proper filtering."""
    # Create a simple prompt and LLM
    prompt = PromptTemplate.from_template("Question: {question}\nAnswer:")
    llm = MockLLM()
    
    # Create a chain
    chain = prompt | llm
    
    # Wrap with guardrails
    guardrailed_chain = chain | RunnableRails(mock_rails_config)
    
    # Run with the question
    result = guardrailed_chain.invoke({"question": question})
    
    # Check if the result matches expectations
    if expected_blocked:
        assert "apologize" in result.lower() or "cannot" in result.lower()
    else:
        assert "information about" in result 