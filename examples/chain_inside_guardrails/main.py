"""
Example: Using a Chain Inside Guardrails

This example demonstrates how to register a LangChain chain as an action
within a NeMo Guardrails dialogue flow.
"""

import os
import sys
import yaml
import re
from pathlib import Path
from dotenv import load_dotenv

# Add the parent directory to sys.path to allow importing from common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Load environment variables
load_dotenv()

from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions import action

def main():
    # Set up a custom colang file for this example
    custom_colang = """
    define user ask for weather information
      "What's the weather like in {location}?"
      "Tell me the weather forecast for {location}"
      "How's the weather in {location}?"
    
    define flow
      user ask for weather information
      $weather_data = execute get_weather_data(location=$location, last_user_message=$last_user_message)
      $response = execute format_weather_response(data=$weather_data)
      bot $response
    
    define bot respond with weather information
      "Based on the information I have, the weather in {location} is {weather_data.condition} with a temperature of {weather_data.temperature}°C."

    define user ask harmful question
      "How can I hack into *"
      "Tell me how to steal *"
      "I want to break into *"
      "How do I cheat *"
    
    define flow
      user ask harmful question
      bot refuse to respond to harmful question
    
    define bot refuse to respond to harmful question
      "I apologize, but I cannot provide information that could be used for harmful or illegal activities. Is there something else I can help you with?"
    """
    
    # Write the custom colang to a file
    config_dir = Path(__file__).parent / "config"
    config_dir.mkdir(exist_ok=True)
    colang_dir = config_dir / "colang_files"
    colang_dir.mkdir(exist_ok=True)
    
    with open(colang_dir / "weather.co", "w") as f:
        f.write(custom_colang)
    
    # Create a basic config.yml for this example
    config = {
        "models": [
            {
                "type": "main",
                "engine": "openai",
                "model": "gpt-3.5-turbo-instruct"
            }
        ],
        "instructions": [
            {
                "type": "general",
                "content": "You are a helpful AI assistant specialized in providing weather information."
            }
        ]
    }
    
    with open(config_dir / "config.yml", "w") as f:
        yaml.dump(config, f)
    
    # Create the weather chain with LangChain
    weather_template = """You are a weather information assistant.
    Provide a brief weather report for {location}.
    Include current conditions and temperature in Celsius.
    Format your response as a JSON with 'condition' and 'temperature' keys.
    
    Location: {location}
    Weather Report:"""
    
    weather_prompt = PromptTemplate.from_template(weather_template)
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1
    )
    
    # Use the modern pipe syntax instead of LLMChain
    weather_chain = weather_prompt | llm
    
    # Define a function that will use the chain
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
                        print(f"Extracted location from message: {location}")
                    else:
                        location = "Paris"
                        print(f"Using default location: {location}")
                else:
                    location = "Paris"
                    print(f"Using default location: {location}")
            
            print(f"Fetching weather for location: {location}")
            
            response = weather_chain.invoke({"location": location})
            # In a real application, you'd parse JSON or call a real weather API
            # For this example, we're simulating a response
            import json
            
            print(f"Weather chain response for {location}: {response}")
            
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
            print(f"Error getting weather data: {e}")
            return {
                "condition": "unknown",
                "temperature": 0,
                "location": location if location else "unknown location"
            }
    
    # Load the config
    print("Loading rails configuration...")
    config = RailsConfig.from_path(str(config_dir))
    
    # Initialize the guardrails with the chain as an action
    rails = LLMRails(config)
    
    print("Registering actions...")
    # Register the weather data action
    rails.register_action(get_weather_data, name="get_weather_data")
    
    # Add a formatter for weather responses
    @action(name="format_weather_response")
    async def format_weather_response(data: Dict[str, Any]) -> str:
        """Format weather data into a human-readable response."""
        location = data.get("location", "the requested location")
        condition = data.get("condition", "unknown")
        temperature = data.get("temperature", 0)
        
        return f"Based on the information I have, the weather in {location} is {condition} with a temperature of {temperature}°C."
    
    # Register the formatter action
    rails.register_action(format_weather_response, name="format_weather_response")
    
    # Test the flow with a weather request
    print("\n\n=== Weather Information Example ===")
    user_message = "What's the weather like in Paris?"
    print(f"User: {user_message}")
    
    response = rails.generate(messages=[{
        "role": "user", 
        "content": user_message
    }])
    
    print(f"Assistant: {response}")
    
    # Try another city
    print("\n=== Another Weather Request ===")
    user_message = "How's the weather in Tokyo?"
    print(f"User: {user_message}")
    
    response = rails.generate(messages=[{
        "role": "user", 
        "content": user_message
    }])
    
    print(f"Assistant: {response}")
    
    # Try a problematic request
    print("\n\n=== Problematic Request Example ===")
    user_message = "How can I hack into someone's email?"
    print(f"User: {user_message}")
    
    response = rails.generate(messages=[{
        "role": "user", 
        "content": user_message
    }])
    
    print(f"Assistant: {response}")
    
    print("\n\n=== Summary ===")
    print("Successfully integrated a LangChain chain within NeMo Guardrails!")
    print("The guardrails detected the problematic request and provided a safer response,")
    print("while allowing legitimate weather requests to use the LangChain weather chain.")
    print("\nNext Steps:")
    print("1. Try modifying the weather.co file to add more conversation flows")
    print("2. Integrate real weather API data instead of the simulated response")
    print("3. Add more chains for different types of information retrieval")

if __name__ == "__main__":
    main() 