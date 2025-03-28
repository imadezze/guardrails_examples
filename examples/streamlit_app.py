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
import asyncio
from openai import OpenAIError
from typing import Dict, List, Any

# Add the parent directory to sys.path to allow importing from examples
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import necessary libraries for the examples
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.actions import action
from langchain.schema.output_parser import OutputParserException
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

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

# Helper function to extract text content from response
def extract_content(response):
    """Extract the text content from various response formats"""
    if response is None:
        return "No response generated."
    
    if isinstance(response, str):
        return response
    
    # Handle dictionary responses
    if isinstance(response, dict):
        # Check for content field directly
        if 'content' in response:
            return response['content'] if response['content'] else "Empty content."
        
        # Check for message format with role and content
        if 'role' in response and 'content' in response:
            return response['content'] if response['content'] else "Empty content."
        
        # Handle message list format that might be returned
        if 'messages' in response and isinstance(response['messages'], list) and len(response['messages']) > 0:
            last_message = response['messages'][-1]
            if isinstance(last_message, dict) and 'content' in last_message:
                return last_message['content']
        
        # If we get here, try to stringify the dict
        return str(response)
    
    # Try to parse JSON strings
    if isinstance(response, str):
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                return extract_content(parsed)  # Recursively process the parsed dict
        except json.JSONDecodeError:
            # Not JSON, return as is
            pass
    
    # Return as is for any other type
    return str(response)

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
    "chain_with_guardrails": "Implements a comprehensive conversation chain with RAG for space exploration",
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
                    
                    # Use st.rerun() instead of experimental_rerun
                    st.rerun()
                
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
        # Use st.rerun() instead of experimental_rerun
        st.rerun()

# Chain With Guardrails example
elif example == "chain_with_guardrails":
    # Define the sample document for knowledge base
    SAMPLE_DOCUMENT = """
    # Space Exploration
    
    Space exploration is the use of astronomy and space technology to explore outer space. 
    While the exploration of space is currently carried out mainly by astronomers with telescopes, 
    its physical exploration is conducted both by uncrewed robotic space probes and human spaceflight.
    
    ## The Moon
    
    The Moon is Earth's only natural satellite. It orbits at an average distance of 384,400 km, 
    about 30 times Earth's diameter. The Moon always presents the same face to Earth, 
    because gravitational pull has locked its rotation to its orbital period.
    
    ## Mars
    
    Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System, 
    larger only than Mercury. In the English language, Mars is named for the Roman god of war.
    Mars is a terrestrial planet with a thin atmosphere, and has a crust primarily composed of 
    elements similar to Earth's crust, as well as a core made of iron and nickel.
    
    ## Jupiter
    
    Jupiter is the fifth planet from the Sun and the largest in the Solar System. 
    It is a gas giant with a mass more than two and a half times that of all the other 
    planets in the Solar System combined, and slightly less than one one-thousandth 
    the mass of the Sun.
    """
    
    # Define toxicity check actions
    @action(name="check_input_toxicity")
    async def check_input_toxicity(user_input: str = "") -> Dict[str, bool]:
        """
        Check if the input text is toxic.
        """
        harmful_keywords = [
            "hack", "steal", "illegal", "bomb", "kill", "hurt", "attack",
            "exploit", "cheat", "fraud", "weapon", "violent", "abuse"
        ]
        
        input_lower = user_input.lower()
        is_toxic = any(keyword in input_lower for keyword in harmful_keywords)
        
        return {"is_toxic": is_toxic}
    
    @action(name="check_output_toxicity")
    async def check_output_toxicity(bot_response: str = "") -> Dict[str, bool]:
        """
        Check if the output text is toxic.
        """
        harmful_keywords = [
            "hack", "steal", "illegal", "bomb", "kill", "hurt", "attack",
            "exploit", "cheat", "fraud", "weapon", "violent", "abuse"
        ]
        
        output_lower = bot_response.lower()
        is_toxic = any(keyword in output_lower for keyword in harmful_keywords)
        
        return {"is_toxic": is_toxic}
    
    # User input section
    st.write("### Space Exploration Assistant with RAG")
    st.write("Ask questions about space, planets, and space exploration. Try both in-context questions (about the Moon, Mars, Jupiter) and out-of-context questions to see how guardrails handle them.")
    
    # Use session state to manage the input field value
    if 'rag_input' not in st.session_state:
        st.session_state.rag_input = ""
    
    # Create the text area with the session state value
    user_input = st.text_area("Enter your question:", 
                             value=st.session_state.rag_input,
                             placeholder="Example: What is the Moon?", 
                             height=100,
                             key="rag_input_area")
    
    # Model selection for RAG system
    rag_model = st.sidebar.selectbox(
        "RAG System Model",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"],
        key="rag_model"
    )
    
    # Model selection for guardrails
    guardrails_model = st.sidebar.selectbox(
        "Guardrails Model",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"],
        key="guardrails_model"
    )
    
    # Temperature settings
    rag_temperature = st.sidebar.slider("RAG Temperature", 
                                      min_value=0.0, 
                                      max_value=1.0, 
                                      value=0.3, 
                                      step=0.1,
                                      key="rag_temp")
    
    guardrails_temperature = st.sidebar.slider("Guardrails Temperature", 
                                             min_value=0.0, 
                                             max_value=1.0, 
                                             value=0.7, 
                                             step=0.1,
                                             key="guardrails_temp")
    
    # Initialize conversation history and knowledge base
    if "example" not in st.session_state or st.session_state.example != example:
        st.session_state.rag_conversation_history = []
        st.session_state.example = example
        st.session_state.rag_debug_info = []
    
    # Initialize the knowledge base and RAG components if they don't exist
    if "rag_knowledge_base" not in st.session_state:
        with st.spinner("Setting up knowledge base..."):
            try:
                # Set up the knowledge base
                docs = [Document(page_content=SAMPLE_DOCUMENT)]
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )
                
                split_docs = text_splitter.split_documents(docs)
                
                # Create embeddings and a vector store
                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                db = FAISS.from_documents(split_docs, embeddings)
                
                # Save in session state
                st.session_state.rag_knowledge_base = db
            except Exception as e:
                error_msg = f"Error setting up knowledge base: {str(e)}"
                st.error(error_msg)
                if debug_mode:
                    st.code(traceback.format_exc())
    
    # Load the guardrails configuration
    if "rag_rails" not in st.session_state or st.session_state.example != example:
        with st.spinner("Loading guardrails configuration..."):
            try:
                config_dir = Path(__file__).resolve().parent / example / "config"
                st.session_state.rag_config = RailsConfig.from_path(str(config_dir))
                st.session_state.rag_rails = LLMRails(config=st.session_state.rag_config)
                
                # Register the actions
                st.session_state.rag_rails.register_action(check_input_toxicity, name="check_input_toxicity")
                st.session_state.rag_rails.register_action(check_output_toxicity, name="check_output_toxicity")
            except Exception as e:
                error_msg = f"Error loading guardrails configuration: {str(e)}"
                st.error(error_msg)
                if debug_mode:
                    st.code(traceback.format_exc())
    
    # Display conversation history
    if 'rag_conversation_history' in st.session_state and st.session_state.rag_conversation_history:
        st.write("### Conversation History")
        # Track pairs of messages
        i = 0
        while i < len(st.session_state.rag_conversation_history):
            if i + 1 < len(st.session_state.rag_conversation_history):
                user_msg = st.session_state.rag_conversation_history[i]
                assistant_msg = st.session_state.rag_conversation_history[i+1]
                
                if user_msg['role'] == 'user' and assistant_msg['role'] == 'assistant':
                    st.markdown(f"**You:** {user_msg['content']}")
                    
                    # If we have raw and final responses stored in metadata
                    if 'metadata' in assistant_msg and 'raw_response' in assistant_msg['metadata']:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Raw LLM Response:**")
                            st.info(assistant_msg['metadata']['raw_response'])
                        with col2:
                            st.markdown("**Final Response:**")
                            st.success(assistant_msg['content'])
                            if 'response_type' in assistant_msg['metadata']:
                                st.caption(f"Response Type: {assistant_msg['metadata']['response_type']}")
                    else:
                        # Regular display if no metadata
                        st.markdown(f"**Assistant:** {assistant_msg['content']}")
                    
                    st.markdown("---")
                
                i += 2
            else:
                # Handle odd number of messages
                st.markdown(f"**{st.session_state.rag_conversation_history[i]['role'].title()}:** {st.session_state.rag_conversation_history[i]['content']}")
                st.markdown("---")
                i += 1
    
    # Submit button
    if st.button("Submit", key="rag_submit_button") and user_input:
        # Check if OpenAI API key is provided
        if not openai_api_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
        else:
            with st.spinner("Generating response..."):
                try:
                    # Add user message to conversation history
                    if 'rag_conversation_history' not in st.session_state:
                        st.session_state.rag_conversation_history = []
                    
                    st.session_state.rag_conversation_history.append({"role": "user", "content": user_input})
                    
                    # Create a retriever
                    retriever = st.session_state.rag_knowledge_base.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 3}
                    )
                    
                    # Create a RAG prompt
                    template = """You are a helpful assistant answering questions about space exploration.
                    Answer the question based on the context provided.
                    
                    If the context doesn't contain the information needed to answer the question, or if the question is not
                    based on the context, say that you don't know instead of making up an answer.
                    
                    Context:
                    {context}
                    
                    User Question: {question}
                    
                    Answer:"""
                    
                    prompt = ChatPromptTemplate.from_template(template)
                    
                    # Initialize the LLM
                    llm = ChatOpenAI(
                        model=rag_model,
                        temperature=rag_temperature,
                        openai_api_key=openai_api_key
                    )
                    
                    # Create the RAG chain
                    rag_chain = (
                        RunnableParallel({
                            "context": retriever,
                            "question": RunnablePassthrough()
                        })
                        | prompt
                        | llm
                    )
                    
                    # Get the raw RAG response
                    rag_response = rag_chain.invoke(user_input)
                    raw_content = rag_response.content if hasattr(rag_response, 'content') else str(rag_response)
                    
                    # Check if the RAG response is "I don't know" or equivalent
                    is_knowledge_gap = any(phrase in raw_content.lower() for phrase in ["i don't know", "don't have information", "cannot answer"])
                    
                    # Check for toxicity
                    toxicity_result = asyncio.run(check_output_toxicity(raw_content))
                    st.session_state.rag_debug_info.append(f"Is content toxic: {toxicity_result['is_toxic']}")
                    
                    # Apply guardrails based on response analysis
                    if toxicity_result["is_toxic"]:
                        # If toxic, let guardrails sanitize it
                        st.session_state.rag_debug_info.append("Content is toxic, using guardrails to sanitize")
                        guardrailed_response = st.session_state.rag_rails.generate(
                            messages=[{"role": "user", "content": user_input}]
                        )
                        # Extract content from guardrailed response
                        final_response = extract_content(guardrailed_response)
                        response_type = "Sanitized (Toxic Content)"
                    elif is_knowledge_gap:
                        # If RAG can't answer, use guardrails for fallback
                        st.session_state.rag_debug_info.append("Knowledge gap detected, using guardrails to provide response")
                        guardrailed_response = st.session_state.rag_rails.generate(
                            messages=[{"role": "user", "content": user_input}]
                        )
                        # Extract content from guardrailed response
                        final_response = extract_content(guardrailed_response)
                        response_type = "Guardrails (Knowledge Gap)"
                    else:
                        # If RAG response is good, preserve it directly
                        st.session_state.rag_debug_info.append("Response is good, preserving RAG content directly")
                        # Since we can't set context variables, we'll just use the raw content directly
                        final_response = raw_content
                        response_type = "RAG (Direct Content)"
                    
                    # Store original guardrailed response for debugging
                    original_guardrailed_response = guardrailed_response if 'guardrailed_response' in locals() else None
                    
                    # Add response to conversation history with metadata
                    st.session_state.rag_conversation_history.append({
                        "role": "assistant", 
                        "content": final_response,
                        "metadata": {
                            "raw_response": raw_content,
                            "response_type": response_type,
                            "original_guardrailed_response": original_guardrailed_response
                        }
                    })
                    
                    # Create columns for layout - for current response
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Raw LLM Response")
                        st.info(raw_content)
                        
                    with col2:
                        st.subheader("Final Response")
                        st.success(final_response)
                        st.caption(f"Response Type: {response_type}")
                    
                    # Show debug information
                    if debug_mode and st.session_state.rag_debug_info:
                        with st.expander("Debug Information"):
                            for info in st.session_state.rag_debug_info:
                                st.text(info)
                            
                            # Show guardrails response format if applicable
                            if 'guardrailed_response' in locals():
                                st.subheader("Guardrails Response Format")
                                st.text(f"Type: {type(guardrailed_response)}")
                                st.text(f"Value: {guardrailed_response}")
                            
                            # Show retrieved documents
                            st.subheader("Retrieved Documents")
                            retrieved_docs = retriever.invoke(user_input)
                            for i, doc in enumerate(retrieved_docs):
                                st.markdown(f"**Document {i+1}:**")
                                st.code(doc.page_content)
                    
                    # Reset debug info for next query
                    st.session_state.rag_debug_info = []
                    
                    # Use st.rerun() to update the conversation history display
                    st.rerun()
                
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
        if not openai_api_key and st.button("Submit", key="rag_api_key_reminder_button"):
            st.error("Please enter your OpenAI API key in the sidebar.")
    
    # Add examples for the user to try
    with st.expander("Try these example questions"):
        example_questions = [
            "What is the Moon?",
            "How far is the Moon from Earth?",
            "What is Mars made of?",
            "How big is Jupiter?",
            "What is the theory of relativity?",
            "Who invented the internet?",
            "Is the Moon made of cheese?",
            "How do I hack a website?"
        ]
        
        # Create two columns to display example questions in two rows
        col1, col2 = st.columns(2)
        
        for i, q in enumerate(example_questions):
            # Alternate between columns
            with col1 if i % 2 == 0 else col2:
                if st.button(q, key=f"example_{q}"):
                    # Directly update the session state value
                    st.session_state.rag_input = q
                    # Add rerun to update the UI immediately with one click
                    st.rerun()
    
    # Clear conversation button
    if st.button("Clear Conversation", key="rag_clear_conversation") and 'rag_conversation_history' in st.session_state:
        st.session_state.rag_conversation_history = []
        # Also clear the input field
        st.session_state.rag_input = ""
        st.rerun()

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