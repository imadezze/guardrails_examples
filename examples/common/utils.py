"""
Common utilities for NeMo Guardrails with LangChain examples.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

def create_example_config(example_dir: Path, custom_colang_files: Optional[Dict[str, str]] = None):
    """
    Create a config directory for an example with optional custom colang files.
    
    Args:
        example_dir: Path to the example directory
        custom_colang_files: Dictionary where keys are filenames and values are file contents
    
    Returns:
        Path to the config directory
    """
    # Create config directory
    config_dir = example_dir / "config"
    config_dir.mkdir(exist_ok=True)
    
    # Create colang_files directory
    colang_dir = config_dir / "colang_files"
    colang_dir.mkdir(exist_ok=True)
    
    # Common config path
    common_config_path = example_dir.parent / "common" / "config"
    
    # Copy config.yml from common directory
    with open(config_dir / "config.yml", "w") as f:
        with open(common_config_path / "config.yml", "r") as src:
            f.write(src.read())
    
    # Copy toxicity.co from common directory
    with open(colang_dir / "toxicity.co", "w") as f:
        with open(common_config_path / "colang_files" / "toxicity.co", "r") as src:
            f.write(src.read())
    
    # Write custom colang files if provided
    if custom_colang_files:
        for filename, content in custom_colang_files.items():
            with open(colang_dir / filename, "w") as f:
                f.write(content)
    
    return config_dir

def check_environment():
    """
    Check if necessary environment variables are set.
    
    Returns:
        bool: True if all necessary environment variables are set
    """
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set them in .env file or export them directly.")
        return False
    
    return True

def format_conversation(messages: List[Dict[str, Any]]) -> str:
    """
    Format a conversation for display.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
    
    Returns:
        str: Formatted conversation
    """
    conversation = []
    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        conversation.append(f"{role}: {content}")
    
    return "\n".join(conversation) 