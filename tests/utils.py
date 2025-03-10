"""
Test utilities for NeMo Guardrails tests.
"""

from typing import Dict, Any, List, Optional, Union, ClassVar
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import Generation, LLMResult
from langchain_core.runnables import Runnable
from pydantic import Field, PrivateAttr


class MockResponse:
    """Mock response for testing."""
    
    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        self.content = content
        self.metadata = metadata or {}
    
    def __str__(self):
        return self.content


class MockLLM(LLM):
    """
    Mock LLM for testing without making API calls.
    
    This class simulates LLM responses based on input patterns and implements
    the LangChain LLM interface for compatibility with chains.
    """
    
    # Private attributes to store data
    _responses: Dict[str, str] = PrivateAttr(default_factory=dict)
    _default_responses: Dict[str, str] = PrivateAttr(default_factory=dict)
    _call_history: List[Dict[str, Any]] = PrivateAttr(default_factory=list)
    
    def __init__(self, responses: Optional[Dict[str, str]] = None, **kwargs):
        """Initialize the mock LLM."""
        super().__init__(**kwargs)
        self._responses = responses or {}
        self._default_responses = {
            "What is the capital of France?": "Paris is the capital of France.",
            "Tell me about artificial intelligence.": "Artificial Intelligence (AI) refers to computer systems designed to perform tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.",
        }
        self._call_history = []
    
    def _call(self, prompt: str, **kwargs) -> str:
        """Process the prompt and return a response."""
        self._call_history.append({"prompt": prompt, "kwargs": kwargs})
        
        # Check if we have a predefined response
        if prompt in self._responses:
            return self._responses[prompt]
        
        # Check default responses
        for key, response in self._default_responses.items():
            if key.lower() in prompt.lower():
                return response
        
        # Check for unsafe content
        unsafe_keywords = ["hack", "illegal", "steal", "attack", "exploit"]
        if any(keyword in prompt.lower() for keyword in unsafe_keywords):
            return "I cannot assist with that request as it appears to involve potentially harmful activities."
        
        # Default response
        return "I'm a helpful AI assistant. How can I assist you today?"
    
    def invoke(self, input_data: Union[str, Dict[str, Any]], *args, **kwargs) -> Union[str, Dict[str, str]]:
        """Simulate a chain invoke call."""
        self._call_history.append({"input": input_data, "args": args, "kwargs": kwargs})
        
        if isinstance(input_data, str):
            return self._call(input_data, **kwargs)
            
        # Handle different input formats
        if isinstance(input_data, dict):
            if "question" in input_data:
                return {"text": self._call(input_data["question"], **kwargs)}
            
            if "messages" in input_data and isinstance(input_data["messages"], list):
                # Get the last user message
                user_message = None
                for message in reversed(input_data["messages"]):
                    if message.get("role") == "user":
                        user_message = message.get("content", "")
                        break
                
                if user_message:
                    return {"response": self._call(user_message, **kwargs)}
            
        return {"output": "Default response from mock LLM chain."}
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "mock_llm"
    
    # Accessors for private attributes
    @property
    def responses(self) -> Dict[str, str]:
        """Get the responses dictionary."""
        return self._responses
    
    @property
    def default_responses(self) -> Dict[str, str]:
        """Get the default responses dictionary."""
        return self._default_responses
    
    @property
    def call_history(self) -> List[Dict[str, Any]]:
        """Get the call history."""
        return self._call_history


class MockToxicityChecker:
    """Mock toxicity checker for testing."""
    
    def __init__(self, toxic_phrases: Optional[List[str]] = None):
        self.toxic_phrases = toxic_phrases or ["hack", "illegal", "steal", "attack", "exploit"]
        self.call_history = []
    
    def check_input_toxicity(self, input_text: str) -> Dict[str, bool]:
        """Check if input contains toxic content."""
        self.call_history.append({"method": "check_input_toxicity", "input": input_text})
        is_toxic = any(phrase in input_text.lower() for phrase in self.toxic_phrases)
        return {"is_toxic": is_toxic}
    
    def check_output_toxicity(self, output_text: str) -> Dict[str, bool]:
        """Check if output contains toxic content."""
        self.call_history.append({"method": "check_output_toxicity", "output": output_text})
        is_toxic = any(phrase in output_text.lower() for phrase in self.toxic_phrases)
        return {"is_toxic": is_toxic} 