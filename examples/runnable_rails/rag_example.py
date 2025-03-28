"""
RAG Example: Integrating NeMo Guardrails with LangChain RAG Pipeline

This example demonstrates how to selectively apply guardrails to the LLM component
of a Retrieval-Augmented Generation (RAG) pipeline.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to allow importing from common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set DEBUG flag
DEBUG = os.environ.get("DEBUG", "").lower() in ("true", "1", "yes", "y")

# Verify API key is loaded (will print redacted version)
openai_api_key = os.environ.get("OPENAI_API_KEY", "")
if openai_api_key:
    print(f"OpenAI API key loaded: {openai_api_key[:4]}...{openai_api_key[-4:]}")
else:
    print("WARNING: OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Import for simplified document retrieval
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import NeMo Guardrails
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.actions import action
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

def setup_guardrails():
    """Set up a basic configuration for guardrails."""
    config_dir = Path(__file__).parent / "rag_config"
    config_dir.mkdir(exist_ok=True)
    
    # Create Colang files directory
    colang_dir = config_dir / "colang_files"
    colang_dir.mkdir(exist_ok=True)
    
    # Create a Colang file for fact checking
    with open(colang_dir / "fact_checking.co", "w") as f:
        f.write("""
define flow fact check response
  $has_hallucination = execute check_for_hallucinations(generation=$generation, context=$context)
  if $has_hallucination
    bot respond with factual limitation
    stop
  
define bot respond with factual limitation
  "I don't have enough information in the provided context to answer that question with certainty. I can only tell you what's mentioned in the available information: {context}"
""")
    
    # Create a minimal config.yml with strong fact-checking instructions
    with open(config_dir / "config.yml", "w") as f:
        f.write("""
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo

instructions:
  - type: general
    content: |
      You are a guardrails-enforced AI assistant that provides ONLY factual information directly found in the context.
      - Never make up information or provide details not present in the context
      - If the context doesn't have the answer, acknowledge the limitation
      - Do not use prior knowledge outside the provided context
      - Avoid speculation and clearly indicate uncertainty
      - Only state what is explicitly mentioned in the context

rails:
  output:
    flows:
      - fact check response
        """)
    
    return RailsConfig.from_path(str(config_dir))

def create_sample_documents():
    """Create intentionally limited sample documents for the RAG example."""
    return [
        "Machine Learning is a subset of AI that enables systems to learn from data.",
        "Neural networks are a type of machine learning model inspired by the human brain.",
        "Deep Learning is based on artificial neural networks with multiple layers.",
        "Supervised learning uses labeled data to train models.",
        "Unsupervised learning works with unlabeled data to find patterns."
    ]
    # Note: Deliberately NOT including when machine learning was invented or by whom

# Add a hallucination detection action
@action(name="check_for_hallucinations")
def check_for_hallucinations(generation, context):
    """Check if the generation contains information not found in the context."""
    # This is a simplified hallucination detector
    # In a real system, this would be much more sophisticated
    
    # List of phrases indicating specific factual claims that would need context support
    factual_indicators = [
        "was invented by", "was created by", "was developed by",
        "was first", "in the year", "was born", "originated in",
        "specific date", "was founded", "was established"
    ]
    
    # Check if generation has factual claims not supported by context
    has_unsupported_claims = False
    for indicator in factual_indicators:
        if indicator in generation.lower() and indicator not in context.lower():
            has_unsupported_claims = True
            print(f"DEBUG: Potential hallucination detected - '{indicator}' not in context")
            break
            
    return has_unsupported_claims

def main():
    print("\n" + "=" * 50)
    print(" RAG Example with Guardrails ".center(50, "="))
    print("=" * 50)
    
    # Setup sample documents and vectorstore
    documents = create_sample_documents()
    
    # Split text into chunks for vectorstore
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    splits = text_splitter.create_documents(documents)
    
    # Create embeddings and vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    
    # Create retriever
    retriever = vectorstore.as_retriever()
    
    # Format the document results
    def format_docs(docs):
        return "\n".join(doc.page_content for doc in docs)
    
    # Create a RAG prompt that explicitly asks for only facts from context
    rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant. Answer ONLY based on the following context:

Context:
{context}

If the context doesn't contain enough information to answer the question fully and accurately, 
say that you don't have enough information in the context.

Question: {question}

Answer:
    """)
    
    # Create LLM
    llm = ChatOpenAI(temperature=0)
    
    # Setup guardrails
    guardrails_config = setup_guardrails()
    guardrails = RunnableRails(guardrails_config)
    
    # Register the hallucination detection action with the guardrails
    rails = LLMRails(guardrails_config)
    rails.register_action(check_for_hallucinations)
    
    print("\n📚 Creating two different RAG chains:")
    print("  1. Standard RAG chain (no guardrails)")
    print("  2. RAG chain with guardrails around just the LLM")
    
    # Standard RAG chain without guardrails
    standard_rag_chain = (
        {
            "context": retriever | format_docs, 
            "question": RunnablePassthrough()
        }
        | rag_prompt 
        | llm 
        | StrOutputParser()
    )
    
    # Test with normal question first
    question = "What is machine learning?"
    print("\n📝 Testing with question: What is machine learning?")
    print("-" * 50)
    
    context = format_docs(retriever.get_relevant_documents(question))
    
    print("\n🔎 Context retrieved:")
    print("-" * 30)
    print(context)
    
    print("\n🔍 Standard RAG chain result:")
    print("-" * 30)
    standard_result = standard_rag_chain.invoke(question)
    print(standard_result)
    
    # For guardrailed responses, we'll use the LLMRails approach which allows us to register the action
    def get_guardrailed_response(question):
        context = format_docs(retriever.get_relevant_documents(question))
        messages = [
            {"role": "system", "content": f"""
You are a helpful AI assistant. Answer ONLY based on the following context:

Context:
{context}

If the context doesn't contain enough information to answer the question fully and accurately, 
say that you don't have enough information in the context.
            """},
            {"role": "user", "content": question}
        ]
        
        # Pass context in the options parameter
        options = {
            "output_vars": True,
            "log": {
                "activated_rails": DEBUG,
                "llm_calls": DEBUG,
                "internal_events": DEBUG
            },
            "context": context  # Pass context here
        }
        
        result = rails.generate(messages=messages, options=options)
        
        # Extract just the content for clean output
        if isinstance(result, dict) and "content" in result:
            response_content = result["content"]
        elif isinstance(result, list) and result and isinstance(result[0], dict) and "content" in result[0]:
            response_content = result[0]["content"]
        else:
            response_content = str(result)
            
        # If in debug mode, also print detailed information about the guardrails
        if DEBUG and hasattr(result, "log") and hasattr(result.log, "activated_rails"):
            print("\n🔍 DEBUG: Activated Rails:")
            for rail in result.log.activated_rails:
                print(f"  - {rail.type}: {rail.name}")
                if hasattr(rail, "decisions"):
                    print(f"    Decisions: {rail.decisions}")
                if hasattr(rail, "executed_actions") and rail.executed_actions:
                    print("    Executed Actions:")
                    for action in rail.executed_actions:
                        print(f"      - {action.action_name}")
                        
        return response_content
    
    print("\n🛡️ Guardrailed RAG chain result:")
    print("-" * 30)
    guardrailed_result = get_guardrailed_response(question)
    print(guardrailed_result)
    
    # Test with a question that will likely cause hallucination
    question = "Who invented machine learning and when?"
    
    print("\n📝 Testing with question that should cause hallucination:")
    print("   'Who invented machine learning and when?'")
    print("-" * 50)
    
    context = format_docs(retriever.get_relevant_documents(question))
    
    print("\n🔎 Context retrieved:")
    print("-" * 30)
    print(context)
    
    print("\n🔍 Standard RAG chain result:")
    print("-" * 30)
    standard_result = standard_rag_chain.invoke(question)
    print(standard_result)
    
    print("\n🛡️ Guardrailed RAG chain result:")
    print("-" * 30)
    guardrailed_result = get_guardrailed_response(question)
    print(guardrailed_result)
    
    # Test with another hallucination-prone question
    question = "When did deep learning become popular?"
    
    print("\n📝 Testing with another hallucination-prone question:")
    print("   'When did deep learning become popular?'")
    print("-" * 50)
    
    context = format_docs(retriever.get_relevant_documents(question))
    
    print("\n🔎 Context retrieved:")
    print("-" * 30)
    print(context)
    
    print("\n🔍 Standard RAG chain result:")
    print("-" * 30)
    standard_result = standard_rag_chain.invoke(question)
    print(standard_result)
    
    print("\n🛡️ Guardrailed RAG chain result:")
    print("-" * 30)
    guardrailed_result = get_guardrailed_response(question)
    print(guardrailed_result)
    
    print("\n📋 Key Takeaways:")
    print("  - RunnableRails allows selective application of guardrails")
    print("  - Guardrails can detect and prevent hallucinations in RAG systems")
    print("  - Standard RAG may generate plausible but unverified information")
    print("  - Guardrailed responses acknowledge limitations when facts are missing")

if __name__ == "__main__":
    main() 