"""
Example: Chain-With-Guardrails for RAG

This example demonstrates how to implement a comprehensive, end-to-end 
Retrieval-Augmented Generation (RAG) conversation chain with guardrails.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import asyncio
from langchain.schema import Document

# Add the parent directory to sys.path to allow importing from common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Load environment variables
load_dotenv()

from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from nemoguardrails.actions import action
from typing import Dict, Any

# Create a sample document for our knowledge base
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
    
    print(f"Input toxicity check: '{user_input}' -> is_toxic: {is_toxic}")
    
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
    
    print(f"Output toxicity check: '{bot_response}' -> is_toxic: {is_toxic}")
    
    return {"is_toxic": is_toxic}

def main():
    # Set up the knowledge base
    docs = [Document(page_content=SAMPLE_DOCUMENT)]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    split_docs = text_splitter.split_documents(docs)
    
    # Create embeddings and a vector store
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(split_docs, embeddings)
    
    # Create a retriever
    retriever = db.as_retriever(
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
        model="gpt-4o-mini",
        temperature=0.3
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
    
    # Create a custom colang file to preserve RAG content
    config_dir = Path(__file__).parent / "config"
    config_dir.mkdir(exist_ok=True)
    colang_dir = config_dir / "colang_files"
    colang_dir.mkdir(exist_ok=True)
    
    # Write custom colang file
    custom_colang = """
    define user ask question
      "What is *"
      "Tell me about *"
      "How does *"
      "Who is *"
      "When *"
      "Where *"
      "Why *"
    
    define flow preserve rag content
      when $preserve_content is True
      bot respond with preserved content
      
    define bot respond with preserved content
      $last_assistant_message
    
    # Include toxicity detection but preserve content when safe
    define flow check output toxicity
      $output_toxicity = execute check_output_toxicity(bot_response=$bot_message)
      if $output_toxicity.is_toxic
        bot respond with sanitized output
        stop
      elif $preserve_content is True
        bot respond with preserved content
        stop
      else
        pass
    """
    
    with open(colang_dir / "preserve_rag.co", "w") as f:
        f.write(custom_colang)
    
    # Create a config.yml file
    with open(config_dir / "config.yml", "w") as f:
        f.write("""
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo-instruct

rails:
  input:
    flows:
      - check input toxicity
  output:
    flows:
      - check output toxicity
      - preserve rag content
    
instructions:
  - type: general
    content: |
      You are a helpful AI assistant that provides accurate and ethical information.
      When you don't know something, admit it rather than making up an answer.
      When a response from a knowledge base is provided, preserve it unless it's harmful.
        """)
    
    # Load the guardrails configuration - use our custom config first
    try:
        # Try to load from our custom config
        config = RailsConfig.from_path(str(config_dir))
        print("Using custom guardrails configuration")
    except Exception as e:
        # Fall back to the common config
        print(f"Custom config error: {e}, falling back to common config")
        config_path = Path(__file__).parent.parent / "common" / "config"
        config = RailsConfig.from_path(str(config_path))
    
    # Create LLMRails instance and register actions
    llm_rails = LLMRails(config)
    llm_rails.register_action(check_input_toxicity, name="check_input_toxicity")
    llm_rails.register_action(check_output_toxicity, name="check_output_toxicity")
    
    # Function to apply guardrails to LLM output
    def apply_guardrails(llm_output, question):
        # Extract the content from the LLM output
        rag_content = llm_output.content
        
        # Check if the RAG response is "I don't know" or equivalent
        is_knowledge_gap = any(phrase in rag_content.lower() for phrase in ["i don't know", "don't have information", "cannot answer"])
        
        # Manual toxicity check
        toxicity_result = asyncio.run(check_output_toxicity(rag_content))
        print(f"Is content toxic: {toxicity_result['is_toxic']}")
        
        if toxicity_result["is_toxic"]:
            # If toxic, let guardrails sanitize it
            print("Content is toxic, using guardrails to sanitize")
            guardrailed_response = llm_rails.generate(
                messages=[
                    {"role": "user", "content": question}
                ]
            )
            return {"output": guardrailed_response}
        elif is_knowledge_gap:
            # If the RAG doesn't know, let guardrails try to answer
            print("Knowledge gap detected, using guardrails to provide response")
            guardrailed_response = llm_rails.generate(
                messages=[
                    {"role": "user", "content": question}
                ]
            )
            return {"output": guardrailed_response}
        else:
            # For good, non-toxic RAG responses, preserve the content
            print("Response is good, preserving RAG content")
            # Return the original RAG content with a disclaimer
            return {"output": {"role": "assistant", "content": rag_content}}
    
    # Test the chain with some safe questions
    print("\n--- Safe Questions ---")
    
    safe_questions = [
        "What is the Moon?",
        "Tell me about Mars.",
        "What is space exploration?"
    ]
    
    for question in safe_questions:
        print(f"\nQuestion: {question}")
        # Run the raw RAG chain for comparison
        raw_response = rag_chain.invoke(question)
        print(f"Raw Output: {raw_response.content}")
        
        # Run the guardrailed chain
        guardrailed_response = apply_guardrails(raw_response, question)
        print(f"Guardrailed Output: {guardrailed_response['output']}")
    
    # Test with questions that might be outside the context or potentially unsafe
    print("\n\n--- Challenging Questions ---")
    
    challenging_questions = [
        "What is the theory of relativity?",  # Outside context
        "How can I build a rocket to go to Mars?",  # Potentially unsafe
        "Is the Moon made of cheese?"  # Factually incorrect
    ]
    
    for question in challenging_questions:
        print(f"\nQuestion: {question}")
        # Run the raw RAG chain for comparison
        raw_response = rag_chain.invoke(question)
        print(f"Raw Output: {raw_response.content}")
        
        # Run the guardrailed chain
        try:
            guardrailed_response = apply_guardrails(raw_response, question)
            print(f"Guardrailed Output: {guardrailed_response['output']}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main() 