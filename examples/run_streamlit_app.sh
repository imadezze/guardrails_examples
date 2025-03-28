#!/bin/bash
# Script to run the NeMo Guardrails Streamlit application

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "../venv" ] && [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating a new one..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    # Activate the virtual environment
    if [ -d "../venv" ]; then
        source ../venv/bin/activate
    else
        source venv/bin/activate
    fi
fi

# Check if streamlit is installed
if ! python -c "import streamlit" &> /dev/null; then
    echo "Streamlit is not installed. Installing dependencies..."
    pip install -r requirements.txt
fi

# Run the Streamlit app
echo "Starting the NeMo Guardrails Streamlit app..."
streamlit run streamlit_app.py

# Deactivate the virtual environment when done
deactivate 