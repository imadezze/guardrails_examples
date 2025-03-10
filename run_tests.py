#!/usr/bin/env python
"""
Test runner for NeMo Guardrails examples.

Usage:
    python run_tests.py                  # Run all tests
    python run_tests.py -k guardrails    # Run tests matching 'guardrails'
    python run_tests.py -v               # Run tests with verbose output
    python run_tests.py --cov=examples   # Run tests with coverage
"""

import sys
import os
import subprocess
from pathlib import Path


def main():
    """Run the tests with pytest."""
    # Ensure we're in the project root directory
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)
    
    # Set up test command
    pytest_args = [
        sys.executable,
        "-m", "pytest",
        "-xvs",  # exit on first failure, verbose, disable capture
        "tests/",
    ]
    
    # Add any additional arguments passed to this script
    if len(sys.argv) > 1:
        pytest_args.extend(sys.argv[1:])
    
    # Print the command that will be executed
    command_str = " ".join(pytest_args)
    print(f"Running: {command_str}")
    
    # Run the tests
    result = subprocess.run(pytest_args)
    
    # Return the pytest exit code
    return result.returncode


if __name__ == "__main__":
    sys.exit(main()) 