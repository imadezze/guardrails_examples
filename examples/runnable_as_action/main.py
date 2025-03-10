"""
Example: Runnable-As-Action

This example demonstrates how to register a LangChain Runnable as an action
within a NeMo Guardrails dialogue flow, allowing conditional execution.
"""

import os
import sys
import re
import io
from pathlib import Path
from dotenv import load_dotenv

# Add the parent directory to sys.path to allow importing from common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Load environment variables
load_dotenv()

from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import Runnable
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions import action

def main():
    # Set up a custom colang file for this example
    custom_colang = """
    define user ask for calculation
      "Calculate {expression}"
      "What is {expression}?"
      "Compute {expression}"
    
    define user ask for movie recommendation
      "Can you recommend a movie about {topic}?"
      "What's a good film about {topic}?"
      "Recommend me a movie related to {topic}"
    
    define flow calculation request
      user ask for calculation
      $result = execute extract_and_calculate(user_message=$user_message)
      bot provide calculation result(result=$result)
    
    define flow movie recommendation request
      user ask for movie recommendation
      $movies = execute extract_and_recommend_movies(user_message=$user_message)
      bot provide movie recommendations(movies=$movies)
    
    define bot provide calculation result
      "The result is {result}."
    
    define bot provide movie recommendations
      "Here are some movie recommendations:\n{movies}"
    """
    
    # Write the custom colang to a file
    config_dir = Path(__file__).parent / "config"
    config_dir.mkdir(exist_ok=True)
    colang_dir = config_dir / "colang_files"
    colang_dir.mkdir(exist_ok=True)
    
    with open(colang_dir / "actions.co", "w") as f:
        f.write(custom_colang)
    
    # Copy common config
    common_config_path = Path(__file__).parent.parent / "common" / "config"
    config_path = Path(__file__).parent / "config"
    
    # Create a basic config.yml for this example
    with open(config_path / "config.yml", "w") as f:
        f.write("""
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo-instruct

instructions:
  - type: general
    content: |
      You are a helpful AI assistant that can perform calculations and recommend movies.
      When asked about calculations, you'll compute the result.
      When asked about movies, you'll provide relevant recommendations.
""")
    
    # Create the calculator chain with LangChain
    calculator_template = """You are a math calculator.
    Calculate the following expression and return ONLY the numeric result without explanation or text.
    
    Expression: {expression}
    
    Result: """
    
    calculator_prompt = PromptTemplate.from_template(calculator_template)
    
    calculator_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )
    
    # Use modern pipe syntax
    calculator_chain = calculator_prompt | calculator_llm
    
    # Create the movie recommendation chain
    movie_template = """You are a movie recommendation system.
    Recommend 3 movies related to the given topic. Format your response as a numbered list with each 
    movie on a new line. Include the title and a very brief (one sentence) description.
    
    Topic: {topic}
    
    Recommendations:"""
    
    movie_prompt = PromptTemplate.from_template(movie_template)
    
    movie_llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # Use modern pipe syntax
    movie_chain = movie_prompt | movie_llm
    
    # Load the config
    config = RailsConfig.from_path(str(config_path))
    
    # Initialize the guardrails
    rails = LLMRails(config)
    
    # Define the calculator function
    def calculate_expression(expression: str) -> str:
        """Calculate a mathematical expression."""
        if os.environ.get("DEBUG", "").lower() == "true":
            print(f"DEBUG: Calculating expression: {expression!r}")
            print(f"DEBUG: Expression type: {type(expression)}")
        
        # Check if the expression is still a variable placeholder
        if expression == "$expression":
            if os.environ.get("DEBUG", "").lower() == "true":
                print("WARNING: The expression variable wasn't properly extracted from the user input")
            return "I'm sorry, but it seems that the expression you want me to calculate is missing. Please provide the expression, and I'll calculate the result for you."
            
        # Now try to calculate with the provided expression
        try:
            response = calculator_chain.invoke({"expression": expression})
            # Handle the new response format (AIMessage)
            result = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            if os.environ.get("DEBUG", "").lower() == "true":
                print(f"DEBUG: Calculation result: {result}")
            return result
        except Exception as e:
            if os.environ.get("DEBUG", "").lower() == "true":
                print(f"ERROR: Failed to calculate expression: {str(e)}")
            return f"I'm sorry, I couldn't calculate '{expression}'. Please check if the expression is valid."
    
    # Create a wrapper function to extract the expression directly from the user message
    def extract_and_calculate(user_message: str) -> str:
        """Extract the expression from the user message and calculate it."""
        # Try to extract the expression using regex
        calculate_pattern = r"Calculate\s+(.*)"
        compute_pattern = r"Compute\s+(.*)"
        what_is_pattern = r"What is\s+(.*)\??"
        
        match = re.search(calculate_pattern, user_message, re.IGNORECASE)
        if not match:
            match = re.search(compute_pattern, user_message, re.IGNORECASE)
        if not match:
            match = re.search(what_is_pattern, user_message, re.IGNORECASE)
            
        if match:
            expression = match.group(1).strip()
            if os.environ.get("DEBUG", "").lower() == "true":
                print(f"DEBUG: Extracted expression: {expression!r}")
            return calculate_expression(expression)
        else:
            return "I couldn't understand the expression. Please provide it in the format 'Calculate X'."
    
    # Define the movie recommendation function
    def recommend_movies(topic: str) -> str:
        """Recommend movies based on a topic."""
        if os.environ.get("DEBUG", "").lower() == "true":
            print(f"DEBUG: Recommending movies for topic: {topic!r}")
            print(f"DEBUG: Topic type: {type(topic)}")
        
        # Check if the topic is still a variable placeholder
        if topic == "$topic":
            if os.environ.get("DEBUG", "").lower() == "true":
                print("WARNING: The topic variable wasn't properly extracted from the user input")
            return "I'm sorry, but it seems that the topic you want recommendations for is missing. Please provide a topic, and I'll recommend some movies for you."
            
        # Now try to get movie recommendations
        try:
            response = movie_chain.invoke({"topic": topic})
            # Handle the new response format (AIMessage)
            result = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            if os.environ.get("DEBUG", "").lower() == "true":
                print(f"DEBUG: Movie recommendations: {result}")
            return result
        except Exception as e:
            if os.environ.get("DEBUG", "").lower() == "true":
                print(f"ERROR: Failed to get movie recommendations: {str(e)}")
            return f"I'm sorry, I couldn't get movie recommendations for '{topic}'. Please try a different topic."
    
    # Create a wrapper function to extract the topic directly from the user message
    def extract_and_recommend_movies(user_message: str) -> str:
        """Extract the topic from the user message and recommend movies."""
        # Try to extract the topic using regex
        recommend_pattern = r"recommend (?:a |me |some )?(?:movie|film)(?:s)?(?: related to| about| on) (.*?)[\?]?$"
        good_film_pattern = r"(?:what'?s|what is) a good film about (.*?)[\?]?$"
        can_you_pattern = r"can you recommend a movie about (.*?)[\?]?$"
        
        match = re.search(recommend_pattern, user_message, re.IGNORECASE)
        if not match:
            match = re.search(good_film_pattern, user_message, re.IGNORECASE)
        if not match:
            match = re.search(can_you_pattern, user_message, re.IGNORECASE)
        if not match:
            match = re.search(r"(?:movie|film) about (.*?)[\?]?$", user_message, re.IGNORECASE)
            
        if match:
            topic = match.group(1).strip()
            if os.environ.get("DEBUG", "").lower() == "true":
                print(f"DEBUG: Extracted topic: {topic!r}")
            return recommend_movies(topic)
        else:
            # If extraction fails, let's try a simpler approach - look for the message after "movie" or "film"
            if "movie" in user_message.lower() or "film" in user_message.lower():
                parts = user_message.split("movie", 1) if "movie" in user_message.lower() else user_message.split("film", 1)
                if len(parts) > 1:
                    potential_topic = parts[1].strip()
                    # Remove leading "about", "related to", etc.
                    for prefix in ["about ", "related to ", "on "]:
                        if potential_topic.lower().startswith(prefix):
                            potential_topic = potential_topic[len(prefix):].strip()
                            break
                    # Remove trailing question mark
                    if potential_topic.endswith("?"):
                        potential_topic = potential_topic[:-1].strip()
                    
                    if potential_topic:
                        if os.environ.get("DEBUG", "").lower() == "true":
                            print(f"DEBUG: Extracted topic (fallback): {potential_topic!r}")
                        return recommend_movies(potential_topic)
            
            return "I couldn't understand the topic. Please provide it in the format 'Can you recommend a movie about X?'"
    
    # Register the actions with rails
    rails.register_action(calculate_expression, name="calculate_expression")
    rails.register_action(recommend_movies, name="recommend_movies")
    rails.register_action(extract_and_calculate, name="extract_and_calculate")
    rails.register_action(extract_and_recommend_movies, name="extract_and_recommend_movies")
    
    # Test the calculation flow
    print("\n--- Testing Calculator Flow ---\n")
    
    test_calculations = [
        "Calculate 23 + 45",
        "Calculate sqrt(144) + 10",
        "Calculate (35 * 12) / 7"
    ]
    
    for user_message in test_calculations:
        print(f"User: {user_message}")
        if os.environ.get("DEBUG", "").lower() == "true":
            print(f"DEBUG: Raw message: {user_message}")
        
        # Temporarily redirect stderr to suppress framework messages
        if os.environ.get("DEBUG", "").lower() != "true":
            original_stderr = sys.stderr
            sys.stderr = io.StringIO()
            
        try:
            response = rails.generate(
                messages=[{
                    "role": "user", 
                    "content": user_message
                }],
                options={
                    "output_vars": True,
                    "log": {
                        "activated_rails": True,
                        "llm_calls": True,
                        "internal_events": True
                    }
                }
            )
        finally:
            # Restore stderr
            if os.environ.get("DEBUG", "").lower() != "true":
                sys.stderr = original_stderr
        
        # Format and display a clean response
        if hasattr(response, "output_data") and "result" in response.output_data:
            result = response.output_data["result"]
            print(f"\nAssistant: The result of the calculation is {result}\n")
        else:
            print(f"\nAssistant: {response[0]['content'] if isinstance(response, list) else response}\n")
        
        # Only show debugging information if DEBUG flag is set
        if os.environ.get("DEBUG", "").lower() == "true":
            # Print detailed logs if available
            if hasattr(response, "log") and hasattr(response.log, "activated_rails"):
                print("\nActivated Rails:")
                for rail in response.log.activated_rails:
                    print(f"- {rail.type}: {rail.name}")
                    print(f"  Decisions: {rail.decisions}")
                    if hasattr(rail, "executed_actions") and rail.executed_actions:
                        print("  Executed Actions:")
                        for action in rail.executed_actions:
                            print(f"    - {action.action_name} with params: {action.action_params}")
                            
            # Print output vars if available
            if hasattr(response, "output_data"):
                print("\nOutput Data:")
                for key, value in response.output_data.items():
                    print(f"- {key}: {value}")
    
    # Test the movie recommendation flow
    print("\n--- Testing Movie Recommendation Flow ---\n")
    
    test_topics = [
        "Can you recommend a movie about space exploration?",
        "What's a good film about artificial intelligence?",
        "Recommend me a movie related to time travel"
    ]
    
    for user_message in test_topics:
        print(f"User: {user_message}")
        if os.environ.get("DEBUG", "").lower() == "true":
            print(f"DEBUG: Raw message: {user_message}")
        
        # Temporarily redirect stderr to suppress framework messages
        if os.environ.get("DEBUG", "").lower() != "true":
            original_stderr = sys.stderr
            sys.stderr = io.StringIO()
            
        try:
            response = rails.generate(
                messages=[{
                    "role": "user", 
                    "content": user_message
                }],
                options={
                    "output_vars": True,
                    "log": {
                        "activated_rails": True,
                        "llm_calls": True,
                        "internal_events": True
                    }
                }
            )
        finally:
            # Restore stderr
            if os.environ.get("DEBUG", "").lower() != "true":
                sys.stderr = original_stderr
        
        # Format and display a clean response
        if hasattr(response, "output_data") and "movies" in response.output_data:
            movies = response.output_data["movies"]
            print(f"\nAssistant: Here are some movie recommendations:\n{movies}\n")
        else:
            print(f"\nAssistant: {response[0]['content'] if isinstance(response, list) else response}\n")
        
        # Only show debugging information if DEBUG flag is set
        if os.environ.get("DEBUG", "").lower() == "true":
            # Print detailed logs if available
            if hasattr(response, "log") and hasattr(response.log, "activated_rails"):
                print("\nActivated Rails:")
                for rail in response.log.activated_rails:
                    print(f"- {rail.type}: {rail.name}")
                    print(f"  Decisions: {rail.decisions}")
                    if hasattr(rail, "executed_actions") and rail.executed_actions:
                        print("  Executed Actions:")
                        for action in rail.executed_actions:
                            print(f"    - {action.action_name} with params: {action.action_params}")
                            
            # Print output vars if available
            if hasattr(response, "output_data"):
                print("\nOutput Data:")
                for key, value in response.output_data.items():
                    print(f"- {key}: {value}")

if __name__ == "__main__":
    main() 