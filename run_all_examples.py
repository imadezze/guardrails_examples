#!/usr/bin/env python
"""
Run all NeMo Guardrails with LangChain examples.
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_environment():
    """Check if the environment is correctly set up."""
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set them in .env file or export them directly.")
        return False
    
    return True

def run_example(example_path):
    """Run a specific example."""
    print(f"\n{'=' * 80}")
    print(f"Running example: {example_path.name}")
    print(f"{'=' * 80}")
    
    result = subprocess.run(
        [sys.executable, "main.py"],
        cwd=example_path,
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    
    if result.stderr:
        print("Errors:")
        print(result.stderr)
    
    print(f"{'=' * 80}")
    print(f"Finished example: {example_path.name}")
    print(f"{'=' * 80}\n")

def main():
    """Run all examples."""
    if not check_environment():
        return
    
    examples_dir = Path(__file__).parent / "examples"
    examples = [
        path for path in examples_dir.iterdir() 
        if path.is_dir() and path.name != "common" and (path / "main.py").exists()
    ]
    
    print(f"Found {len(examples)} examples to run.")
    
    for example_path in examples:
        run_example(example_path)
    
    print("All examples completed.")

if __name__ == "__main__":
    main() 