"""
Direct test of the calculator and movie recommendation chains without NeMo Guardrails.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

def main():
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
    
    # Test the calculator chain
    print("\n--- Testing Calculator Chain Directly ---")
    
    calculations = [
        "23 + 45",
        "sqrt(144) + 10",
        "(35 * 12) / 7"
    ]
    
    for calc in calculations:
        print(f"\nCalculating: {calc}")
        response = calculator_chain.invoke({"expression": calc})
        result = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        print(f"Result: {result}")
    
    # Test the movie recommendation chain
    print("\n--- Testing Movie Recommendation Chain Directly ---")
    
    topics = [
        "space exploration",
        "artificial intelligence",
        "time travel"
    ]
    
    for topic in topics:
        print(f"\nRecommending movies about: {topic}")
        response = movie_chain.invoke({"topic": topic})
        result = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        print(f"Recommendations:\n{result}")


if __name__ == "__main__":
    main() 