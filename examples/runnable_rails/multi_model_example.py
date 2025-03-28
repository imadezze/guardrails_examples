"""
Multi-Model Example: Different Guardrails for Different LLMs

This example demonstrates how to use different guardrails for different models
within a LangChain application using RunnableRails.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to allow importing from common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough

# Import NeMo Guardrails
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.actions import action
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

def setup_fact_checking_guardrails():
    """Set up a configuration for fact-checking guardrails."""
    config_dir = Path(__file__).parent / "fact_check_config"
    config_dir.mkdir(exist_ok=True)
    
    # Create a minimal config.yml for fact checking
    with open(config_dir / "config.yml", "w") as f:
        f.write("""
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo

instructions:
  - type: general
    content: |
      You are a precise, factual AI assistant focused on accuracy.
      - Never make claims without evidence
      - Cite reliable sources when possible
      - Express uncertainty when appropriate
      - Focus on well-established facts
      - Maintain objectivity in all responses
        """)
    
    return RailsConfig.from_path(str(config_dir))

def setup_creative_guardrails():
    """Set up a configuration for content moderation guardrails."""
    config_dir = Path(__file__).parent / "creative_config"
    config_dir.mkdir(exist_ok=True)
    
    # Create Colang files directory
    colang_dir = config_dir / "colang_files"
    colang_dir.mkdir(exist_ok=True)
    
    # Create a Colang file for content moderation
    with open(colang_dir / "moderation.co", "w") as f:
        f.write("""
define flow moderate creative content
  $is_harmful = execute check_content_moderation(content=$generation)
  if $is_harmful
    bot respond with appropriate alternative
    stop
  
define bot respond with appropriate alternative
  "I'd be happy to write a creative story, but I need to make sure it's appropriate and positive. Let me provide an alternative that captures the essence in a more family-friendly way. 
  
  [Family-friendly alternative story]"
""")
    
    # Create a minimal config.yml for creative content
    with open(config_dir / "config.yml", "w") as f:
        f.write("""
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo

instructions:
  - type: general
    content: |
      You are a creative and family-friendly AI assistant.
      - Create engaging content suitable for all ages
      - Avoid dark, frightening, or mature themes
      - Use colorful language and vivid descriptions
      - Be entertaining while remaining appropriate
      - Never include harmful or offensive content
      - Transform any problematic requests into positive, wholesome alternatives

rails:
  output:
    flows:
      - moderate creative content
        """)
    
    return RailsConfig.from_path(str(config_dir))

# Add a content moderation action
@action(name="check_content_moderation")
def check_content_moderation(content):
    """Check if content contains inappropriate themes for family-friendly storytelling."""
    # This is a simplified content moderator
    # In a real system, this would use a more sophisticated approach
    
    # List of potentially problematic themes for creative children's content
    problematic_themes = [
        "violent", "violence", "battle", "blood", "bloodlust", "gore",
        "weapon", "fight", "bloody", "kill", "killing", "death", "dead",
        "brutal", "brutality", "slaughter", "war", "warfare", "combat",
        "conflict", "intense", "terrifying", "scary", "horror", "fear", 
        "frighten", "disturbing", "nightmare", "dark themes", "danger"
    ]
    
    # Check if content contains problematic themes
    has_problematic_content = False
    for theme in problematic_themes:
        if theme in content.lower():
            has_problematic_content = True
            print(f"DEBUG: Potentially problematic content detected - '{theme}' found in creative content")
            break
            
    return has_problematic_content

def main():
    # Enable debug mode based on environment variable
    DEBUG = os.environ.get("DEBUG", "").lower() in ("true", "1", "yes", "y")
    
    print("\n" + "=" * 50)
    print(" Multi-Model Example with Different Guardrails ".center(50, "="))
    print("=" * 50)
    print(f"Debug mode: {'Enabled' if DEBUG else 'Disabled'}")
    
    # Create two different guardrails configurations
    fact_check_config = setup_fact_checking_guardrails()
    creative_config = setup_creative_guardrails()
    
    # Create RunnableRails instances with different configs
    fact_check_rails = RunnableRails(fact_check_config)
    
    # Create LLMRails for creative content to use custom action
    creative_rails = LLMRails(creative_config)
    creative_rails.register_action(check_content_moderation)
    
    # Create LLM models - different temperatures for different purposes
    research_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)  # More precise
    creative_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)  # More creative
    
    # Create prompts for different use cases
    research_prompt = ChatPromptTemplate.from_template(
        "Research the following topic and provide factual information: {topic}"
    )
    
    creative_prompt = ChatPromptTemplate.from_template(
        "Write a short, creative story about: {topic}"
    )
    
    # Create research chain with RunnableRails
    research_chain = (
        research_prompt 
        | (fact_check_rails | research_llm)  # Facts guardrails for research
        | StrOutputParser()
    )
    
    # Define a creative chain function that uses LLMRails directly
    def get_creative_response(topic):
        messages = [
            {"role": "system", "content": "You are a creative and family-friendly AI assistant."},
            {"role": "user", "content": f"Write a short, creative story about: {topic}"}
        ]
        
        # For debug mode, include detailed logs
        options = {
            "output_vars": True,
            "log": {
                "activated_rails": DEBUG,
                "llm_calls": DEBUG,
                "internal_events": DEBUG
            }
        }
        
        result = creative_rails.generate(messages=messages, options=options)
        
        # Extract just the content for clean output
        if isinstance(result, dict) and "content" in result:
            return result["content"]
        elif isinstance(result, list) and result and isinstance(result[0], dict) and "content" in result[0]:
            return result[0]["content"]
        return str(result)
    
    # Create a function that uses standard LLM (no guardrails) for comparison
    def get_standard_creative_response(topic):
        messages = [
            {"role": "system", "content": "You are a creative AI assistant."},
            {"role": "user", "content": f"Write a short, creative story about: {topic}"}
        ]
        
        response = creative_llm.invoke(messages)
        return response.content
    
    # Create a branching function that routes to the appropriate chain
    def process_request(request_type, topic):
        if request_type == "research":
            return research_chain.invoke({"topic": topic})
        elif request_type == "creative":
            return get_creative_response(topic)
        else:
            return "Unknown request type"
    
    print("\n📚 Testing with different request types:")
    print("  1. Research request (uses fact-checking guardrails)")
    print("  2. Creative request (uses content moderation guardrails)")
    
    # Test research request
    research_topic = "quantum computing"
    print(f"\n📝 Research request about: {research_topic}")
    print("-" * 50)
    
    research_result = process_request("research", research_topic)
    print(research_result)
    
    # Test creative request with safe topic
    creative_topic = "a friendly alien visiting Earth"
    print(f"\n🎨 Creative request about: {creative_topic}")
    print("-" * 50)
    
    print("Standard AI (no guardrails):")
    print("-" * 30)
    standard_result = get_standard_creative_response(creative_topic)
    print(standard_result)
    
    print("\nGuardrailed AI:")
    print("-" * 30)
    creative_result = process_request("creative", creative_topic)
    print(creative_result)
    
    # Test a potentially problematic creative request
    problematic_topic = "a violent battle between fierce rivals"
    print(f"\n⚠️ Potentially problematic creative request: {problematic_topic}")
    print("-" * 50)
    
    print("Standard AI (no guardrails):")
    print("-" * 30)
    standard_result = get_standard_creative_response(problematic_topic)
    print(standard_result)
    
    print("\nGuardrailed AI:")
    print("-" * 30)
    problematic_result = process_request("creative", problematic_topic)
    print(problematic_result)
    
    # Test with another problematic topic
    dark_topic = "a terrifying nightmare that causes fear and dread"
    print(f"\n⚠️ Another problematic creative request: {dark_topic}")
    print("-" * 50)
    
    print("Standard AI (no guardrails):")
    print("-" * 30)
    standard_result = get_standard_creative_response(dark_topic)
    print(standard_result)
    
    print("\nGuardrailed AI:")
    print("-" * 30)
    dark_result = process_request("creative", dark_topic)
    print(dark_result)
    
    print("\n📋 Key Takeaways:")
    print("  - Different guardrails can be applied to different models")
    print("  - The guardrail config can be tailored to the use case")
    print("  - Content moderation guardrails can transform problematic topics")
    print("  - Research-focused guardrails ensure factual accuracy")
    print("  - Each model+guardrail combination targets a specific purpose")

if __name__ == "__main__":
    main() 