"""
Streamlit frontend for NeMo Guardrails examples.

This app provides a user interface for interacting with the various NeMo Guardrails examples.
"""

import os
import sys
import yaml
import streamlit as st
from pathlib import Path
import traceback
import re
import json
from openai import OpenAIError
from typing import Dict, List, Any

# Add the parent directory to sys.path to allow importing from examples
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import necessary libraries for the examples
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.actions import action
from langchain.schema.output_parser import OutputParserException

# Set page configuration
st.set_page_config(
    page_title="NeMo Guardrails Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to validate and clean API key
def validate_api_key(api_key):
    """Validates and cleans the OpenAI API key"""
    if not api_key:
        return None
    
    # Remove any whitespace, newlines, or quotes
    cleaned_key = api_key.strip().replace('\n', '').replace('\r', '').replace('"', '').replace("'", '')
    
    # Basic validation - OpenAI keys typically start with "sk-" and have a minimum length
    if not cleaned_key.startswith('sk-') or len(cleaned_key) < 20:
        st.sidebar.warning("⚠️ The API key format looks incorrect. OpenAI keys usually start with 'sk-'")
    
    return cleaned_key

# Sidebar for selecting examples
st.sidebar.title("NeMo Guardrails Examples")
example = st.sidebar.selectbox(
    "Select an example to run",
    ["add_guardrails_to_chain", 
     "chain_inside_guardrails",
     "langsmith_integration",
     "runnable_rails",
     "chain_with_guardrails",
     "runnable_as_action"]
)

# Main title
st.title("NeMo Guardrails Demo")
st.markdown("Explore NeMo Guardrails capabilities through interactive examples.")

# Example descriptions
example_descriptions = {
    "add_guardrails_to_chain": "Directly wraps an existing LangChain chain with NeMo Guardrails",
    "chain_inside_guardrails": "Registers a LangChain chain as an action within a guardrails flow",
    "langsmith_integration": "Integrates LangSmith for monitoring and debugging",
    "runnable_rails": "Uses the core interface to wrap LangChain components with Guardrails",
    "chain_with_guardrails": "Implements a comprehensive conversation chain",
    "runnable_as_action": "Registers a LangChain chain as an action to be invoked within a flow"
}

# Show example description
st.markdown(f"### {example}")
st.markdown(example_descriptions[example])
st.markdown("---")

# Get API key
openai_api_key_input = st.sidebar.text_input("OpenAI API Key", type="password")
openai_api_key = validate_api_key(openai_api_key_input)

# Only set the environment variable if we have a valid API key
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    if "OPENAI_API_KEY" in os.environ:
        st.sidebar.success("API key is set! ✅")
else:
    # Clear the environment variable if the key is invalid
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]

# Debug mode toggle
debug_mode = st.sidebar.checkbox("Debug Mode", value=False, help="Enable to show detailed error information")

# Connection test button
if st.sidebar.button("Test API Connection", key="test_connection"):
    if not openai_api_key:
        st.sidebar.error("Please enter an OpenAI API key first.")
    else:
        try:
            # Try a simple completion to test the connection
            test_llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.0,
                openai_api_key=openai_api_key  # Explicitly pass the API key
            )
            test_response = test_llm.invoke("Test connection")
            st.sidebar.success(f"✅ Connection successful! Response: {test_response.content[:50]}...")
        except Exception as e:
            st.sidebar.error(f"❌ Connection failed: {str(e)}")
            if debug_mode:
                st.sidebar.code(traceback.format_exc())

# Add Guardrails to Chain example
if example == "add_guardrails_to_chain":
    # Import the necessary functions for this example
    from add_guardrails_to_chain.main import (
        check_input_toxicity, 
        check_output_toxicity, 
        apply_guardrails
    )
    
    # User input
    user_input = st.text_area("Enter your question:", height=100)
    
    # Model selection
    model = st.sidebar.selectbox(
        "Select LLM Model",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"]
    )
    
    # Temperature
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    # Load the guardrails configuration when session state is empty or when example changes
    if "rails_config" not in st.session_state or st.session_state.example != example:
        with st.spinner("Loading guardrails configuration..."):
            try:
                config_dir = Path(__file__).resolve().parent / example / "config"
                st.session_state.rails_config = RailsConfig.from_path(str(config_dir))
                st.session_state.llm_rails = LLMRails(config=st.session_state.rails_config)
                st.session_state.llm_rails.register_action(check_input_toxicity, name="check_input_toxicity")
                st.session_state.llm_rails.register_action(check_output_toxicity, name="check_output_toxicity")
                st.session_state.example = example
            except Exception as e:
                error_msg = f"Error loading guardrails configuration: {str(e)}"
                st.error(error_msg)
                if debug_mode:
                    st.code(traceback.format_exc())
    
    # Submit button
    if st.button("Submit", key="main_submit_button") and user_input:
        # Check if OpenAI API key is provided
        if not openai_api_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
        else:
            with st.spinner("Generating response..."):
                try:
                    # Set up a basic LangChain chain
                    template = """You are a helpful assistant.

                    Question: {question}
                    
                    Answer:"""
                    
                    prompt = PromptTemplate.from_template(template)
                    
                    # Initialize the LLM with explicit API key
                    llm = ChatOpenAI(
                        model=model,
                        temperature=temperature,
                        request_timeout=60,  # Set a longer timeout
                        openai_api_key=openai_api_key  # Explicitly pass the API key
                    )
                    
                    # Create a chain using the modern pipe syntax
                    chain = prompt | llm
                    
                    # Get the LLM's raw response
                    llm_response = chain.invoke({"question": user_input})
                    
                    # Apply guardrails
                    guardrailed_response = apply_guardrails(
                        user_input, 
                        llm_response.content, 
                        st.session_state.llm_rails
                    )
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Raw LLM Response")
                        st.info(llm_response.content)
                    
                    with col2:
                        st.subheader("Guardrailed Response")
                        st.success(guardrailed_response)
                    
                    # Show additional debug information
                    with st.expander("Debug Information"):
                        st.write("Raw LLM Response Object:", llm_response)
                        
                        # Check if input would be considered toxic
                        is_toxic_input = any(keyword in user_input.lower() for keyword in [
                            "hack", "steal", "illegal", "bomb", "kill", "hurt", "attack",
                            "exploit", "cheat", "fraud", "weapon", "violent", "abuse"
                        ])
                        st.write("Input toxicity check:", "Toxic" if is_toxic_input else "Safe")
                        
                        # Check if output would be considered toxic
                        is_toxic_output = any(keyword in llm_response.content.lower() for keyword in [
                            "hack", "steal", "illegal", "bomb", "kill", "hurt", "attack",
                            "exploit", "cheat", "fraud", "weapon", "violent", "abuse"
                        ])
                        st.write("Output toxicity check:", "Toxic" if is_toxic_output else "Safe")
                
                except OpenAIError as e:
                    st.error(f"OpenAI API Error: {str(e)}")
                    st.warning("This may be due to an invalid API key, rate limits, or service disruption.")
                    if debug_mode:
                        st.code(traceback.format_exc())
                    
                    # Check specifically for authentication issues
                    error_str = str(e).lower()
                    if "authentication" in error_str or "auth" in error_str or "key" in error_str or "bearer" in error_str:
                        st.error("⚠️ This appears to be an authentication issue with your API key.")
                        st.info("Please make sure your API key is correct and has not expired.")
                
                except (ConnectionError, TimeoutError) as e:
                    st.error(f"Connection Error: {str(e)}")
                    st.warning("Please check your internet connection and try again.")
                    if debug_mode:
                        st.code(traceback.format_exc())
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    if debug_mode:
                        st.code(traceback.format_exc())
                    else:
                        st.info("Enable Debug Mode in the sidebar for more details.")
    else:
        if not openai_api_key and st.button("Submit", key="api_key_reminder_button"):
            st.error("Please enter your OpenAI API key in the sidebar.")

# Chain Inside Guardrails example
elif example == "chain_inside_guardrails":
    # Define the actions for the weather guardrails
    @action(name="get_weather_data")
    async def get_weather_data(location: str = "", last_user_message: str = "") -> Dict[str, Any]:
        """Get weather data for a specific location using LangChain."""
        try:
            # Clean up the location by removing $ if present
            location = location.replace("$", "")
            
            # Try to extract location from the user message if not provided
            if not location or location == "location":
                # Extract location from user's message
                if last_user_message:
                    # Simple extraction using regex - in a real app, use NLP
                    location_match = re.search(r"in\s+([A-Za-z\s]+)[\?\.]?", last_user_message)
                    if location_match:
                        location = location_match.group(1).strip()
                        st.session_state.debug_info = st.session_state.get('debug_info', []) + [f"Extracted location from message: {location}"]
                    else:
                        location = "Paris"
                        st.session_state.debug_info = st.session_state.get('debug_info', []) + [f"Using default location: {location}"]
                else:
                    location = "Paris"
                    st.session_state.debug_info = st.session_state.get('debug_info', []) + [f"Using default location: {location}"]
            
            st.session_state.debug_info = st.session_state.get('debug_info', []) + [f"Fetching weather for location: {location}"]
            
            # Create the weather chain
            weather_template = """You are a weather information assistant.
            Provide a brief weather report for {location}.
            Include current conditions and temperature in Celsius.
            Format your response as a JSON with 'condition' and 'temperature' keys.
            
            Location: {location}
            Weather Report:"""
            
            weather_prompt = PromptTemplate.from_template(weather_template)
            
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.1,
                openai_api_key=openai_api_key
            )
            
            # Use the modern pipe syntax
            weather_chain = weather_prompt | llm
            
            response = weather_chain.invoke({"location": location})
            st.session_state.debug_info = st.session_state.get('debug_info', []) + [f"Weather chain response for {location}: {response}"]
            
            # Extract JSON-like content from the response (simplified)
            json_pattern = r'\{.*\}'
            # Update to handle the new response format
            response_text = response.content if hasattr(response, 'content') else str(response)
            match = re.search(json_pattern, response_text, re.DOTALL)
            
            if match:
                try:
                    weather_data = json.loads(match.group(0))
                    # Add the location to the weather data
                    weather_data["location"] = location
                    return weather_data
                except json.JSONDecodeError:
                    # Fallback if JSON parsing fails
                    return {
                        "condition": "sunny",
                        "temperature": 25,
                        "location": location
                    }
            else:
                # Fallback
                return {
                    "condition": "sunny",
                    "temperature": 25,
                    "location": location
                }
        except Exception as e:
            st.session_state.debug_info = st.session_state.get('debug_info', []) + [f"Error getting weather data: {e}"]
            return {
                "condition": "unknown",
                "temperature": 0,
                "location": location if location else "unknown location"
            }
    
    @action(name="format_weather_response")
    async def format_weather_response(data: Dict[str, Any]) -> str:
        """Format weather data into a human-readable response."""
        location = data.get("location", "the requested location")
        condition = data.get("condition", "unknown")
        temperature = data.get("temperature", 0)
        
        return f"Based on the information I have, the weather in {location} is {condition} with a temperature of {temperature}°C."

    # User input section
    st.write("### Weather Information Assistant")
    st.write("Ask about the weather in different locations or try asking harmful questions to see how guardrails work.")
    user_input = st.text_area("Enter your question:", 
                             placeholder="Example: What's the weather like in Tokyo?", 
                             height=100)
    
    # Model selection
    model = st.sidebar.selectbox(
        "Select LLM Model",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"],
        key="weather_model"
    )
    
    # Temperature
    temperature = st.sidebar.slider("Temperature", 
                                   min_value=0.0, 
                                   max_value=1.0, 
                                   value=0.1, 
                                   step=0.1,
                                   key="weather_temp")
    
    # Initialize or reset debug info when loading the example
    if "example" not in st.session_state or st.session_state.example != example:
        st.session_state.debug_info = []
        st.session_state.conversation_history = []
    
    # Load the guardrails configuration when session state is empty or when example changes
    if "weather_rails" not in st.session_state or st.session_state.example != example:
        with st.spinner("Loading guardrails configuration..."):
            try:
                config_dir = Path(__file__).resolve().parent / example / "config"
                st.session_state.weather_config = RailsConfig.from_path(str(config_dir))
                st.session_state.weather_rails = LLMRails(config=st.session_state.weather_config)
                
                # Register the actions
                st.session_state.weather_rails.register_action(get_weather_data, name="get_weather_data")
                st.session_state.weather_rails.register_action(format_weather_response, name="format_weather_response")
                
                st.session_state.example = example
            except Exception as e:
                error_msg = f"Error loading guardrails configuration: {str(e)}"
                st.error(error_msg)
                if debug_mode:
                    st.code(traceback.format_exc())
    
    # Display conversation history
    if 'conversation_history' in st.session_state and st.session_state.conversation_history:
        st.write("### Conversation History")
        for i, entry in enumerate(st.session_state.conversation_history):
            if entry['role'] == 'user':
                st.markdown(f"**You:** {entry['content']}")
            else:
                st.markdown(f"**Assistant:** {entry['content']}")
            st.markdown("---")
    
    # Submit button
    if st.button("Submit", key="weather_submit_button") and user_input:
        # Check if OpenAI API key is provided
        if not openai_api_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
        else:
            with st.spinner("Generating response..."):
                try:
                    # Add user message to conversation history
                    if 'conversation_history' not in st.session_state:
                        st.session_state.conversation_history = []
                    
                    st.session_state.conversation_history.append({"role": "user", "content": user_input})
                    
                    # Generate response using guardrails
                    response = st.session_state.weather_rails.generate(
                        messages=[{"role": "user", "content": user_input}]
                    )
                    
                    # Add response to conversation history
                    st.session_state.conversation_history.append({"role": "assistant", "content": response})
                    
                    # Create columns for layout
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Your Question")
                        st.info(user_input)
                    
                    with col2:
                        st.subheader("Guardrailed Response")
                        st.success(response)
                    
                    # Show debug information
                    if debug_mode and 'debug_info' in st.session_state:
                        with st.expander("Debug Information"):
                            for info in st.session_state.debug_info:
                                st.text(info)
                    
                    # Reset debug info for next query
                    st.session_state.debug_info = []
                    
                    # Force a rerun to update the conversation history display
                    st.experimental_rerun()
                
                except OpenAIError as e:
                    st.error(f"OpenAI API Error: {str(e)}")
                    st.warning("This may be due to an invalid API key, rate limits, or service disruption.")
                    if debug_mode:
                        st.code(traceback.format_exc())
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    if debug_mode:
                        st.code(traceback.format_exc())
                    else:
                        st.info("Enable Debug Mode in the sidebar for more details.")
    else:
        if not openai_api_key and st.button("Submit", key="weather_api_key_reminder_button"):
            st.error("Please enter your OpenAI API key in the sidebar.")
    
    # Clear conversation button
    if st.button("Clear Conversation", key="clear_conversation") and 'conversation_history' in st.session_state:
        st.session_state.conversation_history = []
        st.experimental_rerun()

# Placeholder for other examples
else:
    st.info(f"The {example} example is not yet implemented in the Streamlit interface. Coming soon!")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center">
        <p>NeMo Guardrails Demo | Built with Streamlit | <a href="https://github.com/NVIDIA/NeMo-Guardrails">GitHub Repository</a></p>
    </div>
    """, unsafe_allow_html=True) 