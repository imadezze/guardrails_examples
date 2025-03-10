# NeMo Guardrails with LangChain Integration

This directory contains examples demonstrating how to integrate NeMo Guardrails with LangChain.

## Overview

NeMo Guardrails provides two main approaches for integrating with LangChain:

1. **Using LLMRails directly** - For direct message handling with full guardrails capabilities
2. **Using RunnableRails with LangChain pipelines** - For integrating guardrails into LangChain's Runnable interface

Both approaches allow you to add safety and content filters to your LLM applications, but they have different strengths and use cases.

## Understanding the Two Integration Approaches

### LLMRails Approach (Direct Usage)

**LLMRails** is the core interface of NeMo Guardrails that provides direct control over the guardrails system. When using this approach:

- You interact with the guardrails system through a message-based interface
- You have full control over action registration with `rails.register_action()`
- It provides complete access to detailed logging and all guardrails capabilities
- The input/output format is consistent: you provide messages and get message responses
- Custom actions work reliably as long as they're properly registered

**Example Implementation:**
```python
# Create and configure LLMRails
rails = LLMRails(config)
rails.register_action(check_input_toxicity)
rails.register_action(check_output_toxicity)

# Use directly with messages
messages = [{"role": "user", "content": "Tell me a joke about programming"}]
response = rails.generate(messages=messages, options=logging_options)
```

This approach is more direct, with explicit control over the guardrails system. It's ideal when you need full access to guardrails features like custom actions and detailed logging.

### RunnableRails Approach (LangChain Integration)

**RunnableRails** is a wrapper that implements LangChain's Runnable protocol to make guardrails compatible with LangChain's component-based architecture. Its goal is to:

- Integrate seamlessly with LangChain's Expression Language (LCEL) using the pipe operator (`|`)
- Allow guardrails to be inserted at specific points in a LangChain processing pipeline
- Support LangChain's various input/output formats (prompts, chat histories, completions)
- Enable selective guardrail application to specific components rather than the entire application

**Example Implementation:**
```python
# Create components
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
model = ChatOpenAI()
output_parser = StrOutputParser()

# Create RunnableRails instance
guardrails = RunnableRails(config)

# Create a chain with guardrails around the LLM model
chain = prompt | (guardrails | model) | output_parser

# Use with LangChain's invoke pattern
result = chain.invoke({"topic": "programming"})
```

## The Goal of RunnableRails

The primary goal of RunnableRails is to make NeMo Guardrails a "first-class citizen" in the LangChain ecosystem by:

1. **Supporting the Runnable Protocol**: Implementing LangChain's interface for composable components
2. **Enabling Selective Application**: Allowing guardrails to wrap specific parts of a chain
3. **Format Compatibility**: Supporting LangChain's various input/output formats
4. **Chain Composition**: Making guardrails work with LangChain's pipe operator for building chains
5. **Flexibility**: Giving developers control over where guardrails are applied in complex pipelines

For example, in a RAG application, you might want to apply guardrails only to the final LLM response, but not to the retrieval process:

```python
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | (guardrails | llm)  # Guardrails only around the LLM
    | output_parser
)
```

## Examples

### 1. Simple Example (`simple_main.py`)

This example demonstrates:
- Creating toxicity check actions with proper action decorators
- Setting up a custom guardrails configuration with Colang flows
- Using LLMRails with direct message handling
- Implementing detailed logging for debugging

```bash
python simple_main.py             # Run with standard output
DEBUG=true python simple_main.py  # Run with detailed logging
```

### 2. Main Example (`main.py`)

This more complex example showcases:
- Multiple guardrails implementation techniques
- Custom key mapping for inputs and outputs 
- Streaming support with guardrails
- Various LangChain component integrations

#### Implementation Details:

- **Conditional Debug Logging**: Uses an environment variable `DEBUG` to toggle detailed logging throughout the application
- **Custom Action Registration**: Demonstrates registering custom actions (`check_input_quality` and `check_output_accuracy`) with LLMRails
- **Proper Input Transformation**: Shows the correct way to format inputs for RunnableRails using the required `{"input": messages}` structure
- **Robust Response Formatting**: Implements a comprehensive `format_response` function that handles various response formats, including dictionaries, lists, and different key structures
- **Multiple Integration Approaches**:
  - Basic usage with direct RunnableRails initialization
  - Input transformation approach that properly formats messages for RunnableRails
  - Example of streaming support (demonstrated conceptually)
  - Comprehensive solution using LLMRails directly for more reliable action registration
- **Error Handling**: Implements try-except blocks around each example to ensure graceful error handling
- **Clean Output Formatting**: Ensures clean, user-friendly output when running without debug mode enabled

The implementation follows best practices for integrating NeMo Guardrails with LangChain, with particular attention to the correct initialization of RunnableRails and proper input/output formatting.

```bash
python main.py             # Run with standard output
DEBUG=true python main.py  # Run with detailed debug logging
```

### 3. RAG Example (`rag_example.py`)

The RAG example demonstrates how to apply guardrails specifically to the LLM component within a Retrieval-Augmented Generation pipeline. This allows for selective application of guardrails to just the content generation phase while maintaining the retrieval functionality separate from guardrails.

#### Key Implementation Features:

- **Hallucination Detection**: Custom action to verify generated content against retrieved context
- **Context-Aware Guardrails**: Colang flows that analyze both the generation and its context
- **Selective Application**: Guardrails applied only to the LLM component, not to the retrieval process
- **Detailed Debug Logging**: Configurable logging to trace guardrail activation
- **Format Transformation**: Clean extraction of content from guardrailed responses

#### Example Implementation Overview:

```python
# Define a hallucination check action
@action(name="check_for_hallucinations")
def check_for_hallucinations(generation, context):
    # Compare generation against context to detect unsupported claims
    # Return whether hallucination was detected
    return {"hallucination_detected": False}  # Simplified example

# Create standard RAG chain
standard_rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm  # No guardrails
    | output_parser
)

# Create guardrailed RAG function
def get_guardrailed_response(question):
    # Get context from retriever
    docs = retriever.invoke(question)
    context = format_docs(docs)
    
    # Use LLMRails with context in options
    rails = LLMRails(config)
    rails.register_action(check_for_hallucinations)
    
    messages = [{"role": "user", "content": f"Context: {context}\nQuestion: {question}"}]
    options = {"context": context, "output_vars": True}
    
    result = rails.generate(messages=messages, options=options)
    return extract_content(result)  # Extract just the response content
```

#### Example Output:

When asked a question like "When did deep learning become popular?", the system demonstrates clear differences between guardrailed and non-guardrailed responses:

```
Standard RAG chain response:
Deep learning became popular in the early 2010s, especially after breakthrough 
results in computer vision with convolutional neural networks and the development 
of powerful frameworks like TensorFlow and PyTorch that made implementation more 
accessible.

Guardrailed RAG chain response:
I don't have enough information in the context to provide a specific timeline for 
when deep learning became popular. The provided context mentions neural networks 
and supervised learning concepts, but doesn't contain specific dates about when 
deep learning gained popularity.
```

The standard RAG chain produces a plausible-sounding answer that isn't supported by the context, while the guardrailed version correctly acknowledges the limitation of the available information, preventing potential hallucinations.

#### Response Analysis

The RAG example demonstrates how guardrails can effectively prevent hallucinations in retrieval-augmented generation systems. When analyzing the complete output:

1. **Context Retrieval**: Both approaches use the same context retrieval mechanism, ensuring they start with identical information.

2. **Response Pattern Differences**:
   - **Standard RAG Responses**: Often provide specific details not found in the context, such as dates, names, or technical specifications. While these may sound plausible and confident, they are fabricated.
   - **Guardrailed RAG Responses**: Include phrases like "I don't have enough information in the context" or "Based only on the provided context" to clearly indicate the boundaries of knowledge.

3. **Hallucination Prevention Strategy**:
   - The guardrails actively check if claims in the generation are supported by the retrieved context
   - Synchronous action calls (`check_for_hallucinations`) verify content before delivery
   - When unsupported claims are detected, the response is reformulated to acknowledge information limitations

4. **Technical Implementation**:
   - The custom `check_for_hallucinations` action receives both the generated content and context for comparison
   - Debug logs show when hallucination checks are triggered and their outcomes
   - The guardrail can be configured with different levels of strictness for various use cases

This example clearly illustrates that even the most advanced LLMs can generate unsupported information when using RAG, and demonstrates how guardrails provide an effective solution to this critical problem in knowledge-based applications.

#### Key Takeaways:
- RAG systems are susceptible to hallucinations when the LLM generates information not in the context
- Guardrails can effectively detect and prevent such hallucinations
- The implementation allows precise control over when and how guardrails are applied
- Debug logging provides insights into guardrail activation and decision-making

```bash
python rag_example.py
DEBUG=true python rag_example.py  # Run with detailed logging
```

### 4. Multi-Model Example (`multi_model_example.py`)

The multi-model example demonstrates how different guardrails can be applied to different models based on their specific purpose and content type. This allows for tailored safety measures and content policies for each use case.

#### Key Implementation Features:

- **Purpose-Specific Guardrails**: Different guardrail configurations for different types of content
- **Content Moderation**: Advanced detection of potentially problematic creative content
- **Research vs. Creative Modes**: Different models and guardrails for factual vs. creative requests
- **Direct Comparison**: Side-by-side output showing guardrailed vs. non-guardrailed responses
- **Transformation Logic**: Problematic content transformed into family-friendly alternatives

#### Example Implementation Overview:

```python
# Set up different guardrail configurations
fact_check_config = setup_fact_checking_guardrails()
creative_config = setup_creative_guardrails()

# Create guardrails with different configs
fact_check_rails = RunnableRails(fact_check_config)
creative_rails = LLMRails(creative_config)  # Using LLMRails for custom action

# Register custom content moderation action
@action(name="check_content_moderation")
def check_content_moderation(content):
    # Check for problematic themes in creative content
    problematic_themes = ["violent", "scary", "disturbing", ...]
    
    # Return whether problematic content was detected
    for theme in problematic_themes:
        if theme in content.lower():
            return True
    return False

# Register the action with creative rails
creative_rails.register_action(check_content_moderation)

# Use different models for different purposes
research_llm = ChatOpenAI(temperature=0)   # More precise
creative_llm = ChatOpenAI(temperature=0.7) # More creative

# Function to get guardrailed creative response
def get_creative_response(topic):
    messages = [
        {"role": "system", "content": "You are a creative and family-friendly AI assistant."},
        {"role": "user", "content": f"Write a short, creative story about: {topic}"}
    ]
    
    result = creative_rails.generate(messages=messages, options={"output_vars": True})
    return extract_content(result)  # Extract just the response content
```

#### Example Output:

When asked to write about "a violent battle between fierce rivals", the system shows dramatic differences:

```
Standard AI (no guardrails):
------------------------------
In the kingdom of Aleria, two powerful factions had long been at odds with each 
other - the Phoenix Clan and the Dragon Brotherhood. For centuries, their rivalry 
had simmered and boiled over, resulting in numerous skirmishes and battles.

One fateful day, tensions reached a breaking point as the leaders of both factions, 
the fiery Phoenix Queen Lysandra and the cunning Dragon Lord Draven, declared war 
on each other. The land trembled with anticipation as both armies gathered on the 
field of battle, their weapons gleaming in the sunlight.

The clash was fierce and brutal, with swords clashing, arrows flying, and magic 
crackling in the air. The Phoenix warriors, their armor shining like flames, 
fought with unmatched ferocity, while the Dragon soldiers, their scales glistening 
in the sun, struck with deadly precision.

As the battle raged on, the ground became stained with blood and the air filled 
with the cries of the fallen. Each side fought with unwavering determination, 
fueled by centuries of resentment and hatred.
[... more violent content ...]

Guardrailed AI:
------------------------------
Once upon a time in a whimsical land far, far away, two groups of daring 
adventurers clashed in an epic showdown that shook the very earth beneath their 
feet. The first group, led by the valiant Captain Rosewood, rode in on majestic 
unicorns, their sparkling horns glinting in the sunlight. On the opposing side, 
the brave Viking warrior, Chief Thunderheart, commanded a fleet of flying dragons, 
their fiery breath lighting up the sky.

As the battle commenced, swords clashed and spells were cast, creating a dazzling 
display of colors and lights that danced across the battlefield. Yet, instead of 
anger and hatred, there was a sense of camaraderie and respect between the 
rivals, as they fought not out of malice but for the honor and glory of their 
kingdoms.

In the end, a truce was called, and the two sides joined forces to defend their 
realm against a greater threat, proving that even the fiercest of rivals can find 
common ground in the face of true danger. And so, peace was restored to the land, 
and unity prevailed over discord, reminding all that sometimes, it takes a battle 
to bring people together in harmony and understanding.
```

Similarly, when asked about "a terrifying nightmare", the guardrailed version transforms dark content into a story with a positive resolution and teaching moment.

#### Content Transformation Examples:
- **For the violent battle**: The guardrail transformed graphic violence into a fantasy "clash" with positive resolution and themes of camaraderie
- **For the nightmare scenario**: The guardrail kept some eerie elements but added a positive fairy character and happy ending
- **The family-friendly alien story**: Passed through with minimal changes since the content was already appropriate

#### Response Analysis

The multi-model example demonstrates two distinct types of guardrails:

1. **Research-focused guardrails** (fact-checking) for the research query about quantum computing
2. **Content moderation guardrails** for creative writing tasks

The format of the responses shows clear differences between guardrailed and non-guardrailed content:

**For standard AI (no guardrails):**
- Raw, unfiltered creative content
- Includes potentially problematic elements like violence, fear, and dread
- No transformation of inappropriate themes
- Dark outcomes and negative emotions may be present

**For guardrailed AI:**
- Shows "Synchronous action `check_content_moderation` has been called" indicating the guardrail is actively working
- Content has been modified to be more family-friendly while preserving creativity
- For violent topics: battle scenes are transformed to be less graphic, more fantastical, with positive outcomes
- For nightmare topics: keeps some scary elements but adds a positive resolution (e.g., with the dream fairy)
- Maintains engagement and imagination while removing problematic content

The output also includes technical details like the full response data structure and an `output_data` dictionary with detailed information about the generation process, activated rails, and event tracking.

This clear side-by-side comparison effectively demonstrates how content moderation guardrails can detect potentially problematic content and transform it into more appropriate alternatives while preserving the creative essence of the original request.

#### Key Takeaways Summary:
- Demonstrates the application of different guardrails to different types of content and models
- Shows how content moderation guardrails can transform inappropriate content while preserving creative value
- Illustrates the clear difference between unfiltered and guardrailed responses for problematic topics
- Provides a practical implementation of how to use different guardrail configurations for varied use cases
- Emphasizes how the right guardrail configuration can significantly improve content safety without losing engagement

```bash
python multi_model_example.py
DEBUG=true python multi_model_example.py  # Run with detailed logging
```

### 5. Key Mapping Example (`key_mapping_example.py`)

The Key Mapping Example showcases how to effectively handle custom input/output formats when integrating NeMo Guardrails with existing APIs or systems. It demonstrates four distinct approaches to key mapping that enable seamless integration between various data formats and the guardrails system.

#### Key Implementation Features:

- **Multiple Mapping Approaches**: Four different methods for handling varied input/output formats
- **Format Transformation**: Converting between custom API schemas and guardrails requirements
- **Rails-as-Runnable**: Using guardrails as the primary runnable component in a processing chain
- **Legacy API Integration**: Handling deeply nested legacy API structures
- **Format Preservation**: Maintaining original request metadata in transformed responses

#### The Four Example Approaches:

1. **Default Mapping (No Custom Keys)**:
   - Uses standard LangChain piping with minimal transformation
   - RunnableRails transparently handles message format conversion
   - Ideal for new applications with simple prompt templates
   - Example: `default_chain = prompt | (default_guardrails | llm) | StrOutputParser()`

2. **Custom Input Mapping**:
   - Transforms custom API input formats (e.g., `{"query": "topic"}`) to guardrails format
   - Converts external query parameters to the expected message format
   - Uses the critical `{"input": messages}` format required by RunnableRails
   - Extracts response content from guardrails output structure

3. **Custom Input and Output Mapping**:
   - Handles both directions of mapping with separate functions
   - Transforms complex, nested input parameters to guardrails format
   - Maps guardrails outputs back to custom API response formats
   - Preserves request metadata across the transformation process

4. **Rails-as-Runnable with Legacy API**:
   - Demonstrates guardrails as the primary runnable component
   - Handles deeply nested legacy API formats (e.g., `data.request.parameters.userQuery`)
   - Shows integration with legacy systems without changing their API structure
   - Guardrails component manages LLM interaction internally
   - Preserves backward compatibility while adding guardrails protection

#### Why Key Mapping is Critical

The examples demonstrate several important aspects of key mapping:

1. **Input Format Requirement**: RunnableRails specifically requires the `{"input": messages}` format, not `{"messages": messages}`
2. **Output Structure Understanding**: Guardrails returns output in `{"output": {"role": "assistant", "content": "..."}}` format
3. **Legacy Integration**: Key mapping enables guardrails to work with existing systems without API changes
4. **Flexibility**: Different mapping approaches can be used based on application needs
5. **Error Prevention**: Proper mapping prevents common errors like "No `input` key found in the input dictionary"

#### Implementation Highlights:

```python
# Example 2: Custom Input Mapping - transforms query format to messages
def process_with_custom_input(inputs):
    # Transform input to guardrails format
    messages = input_mapper(inputs)
    # The key insight: guardrails expects an "input" key
    guardrails_output = basic_guardrails.invoke({"input": messages})
    # Extract content from guardrails output structure
    if isinstance(guardrails_output, dict) and "output" in guardrails_output:
        output = guardrails_output["output"]
        if isinstance(output, dict) and "content" in output:
            return output["content"]
    return "No response generated."

# Example 4: Rails-as-Runnable with Legacy API
def rails_as_runnable(legacy_input):
    # Extract from nested legacy format
    messages = legacy_api_mapper(legacy_input)
    # Guardrails as the primary runnable component
    guardrails_output = basic_guardrails.invoke({"input": messages})
    # Extract and transform output
    content = ""
    if isinstance(guardrails_output, dict) and "output" in guardrails_output:
        output = guardrails_output["output"]
        if isinstance(output, dict) and "content" in output:
            content = output["content"]
    # Map back to legacy format
    return legacy_api_output_mapper(content, legacy_input)
```

#### Example Output

The example demonstrates transforming various input formats:

1. Simple topic format: `{"topic": "machine learning"}`
2. Custom API format: `{"query": "artificial intelligence"}`
3. Complex nested format: `{"query_params": {"subject": "quantum computing"}}`
4. Legacy API format:
```json
{
  "version": "1.2.0",
  "data": {
    "request": {
      "parameters": {
        "userQuery": "neural networks"
      }
    },
    "metadata": {
      "requestId": "req-12345-abc"
    }
  }
}
```

And produces corresponding transformed outputs that maintain the expected schema while incorporating guardrailed content.

#### Key Takeaways:

- Proper key mapping is essential for integrating RunnableRails with existing systems
- The `{"input": messages}` format is required by RunnableRails, not `{"messages": messages}`
- Guardrails can serve as a primary runnable component that handles LLM interactions internally
- Format transformation enables guardrails to work with diverse API architectures
- Key mapping allows legacy systems to benefit from guardrails without API changes

```bash
python key_mapping_example.py
```

## When to Use RunnableRails vs. LLMRails

**Use RunnableRails when:**
- You need to integrate with LangChain's component architecture
- You're using built-in rails that don't require custom actions
- You want selective application of guardrails to specific components
- You need to adapt between different input/output formats
- You're building complex pipelines with branching, routing, or multiple models

**Use LLMRails (or the workaround) when:**
- You need custom actions in your guardrails
- You require detailed logging and debugging of guardrails behavior
- You have complex Colang flows with custom behaviors
- You need full control over the guardrails configuration and execution

## Implementation Details

### Action Registration

For guardrails to work correctly, actions must be:

1. Decorated with the `@action` decorator with the exact name matching what's in your Colang flow
2. Explicitly registered with the rail instance using `rails.register_action()`
3. Have parameter signatures that match what the guardrails system expects

Example:
```python
@action(name="check_input_toxicity")
def check_input_toxicity(messages=None):
    # Implementation...
    return {"is_toxic": False, "toxicity_score": 0.1}

# Later in code:
rails = LLMRails(config)
rails.register_action(check_input_toxicity)
```

### Colang Flow Definitions

Guardrails use Colang flows to define when and how actions are triggered:

```
define flow check input toxicity
  $toxicity = execute check_input_toxicity(messages=$messages)
  if $toxicity.is_toxic
    bot refuse to respond due to harmful content
    stop
```

### Configuration Structure

A proper NeMo Guardrails configuration requires:
- Model configuration (type, engine, model name)
- Instructions for the assistant
- Rails definitions (input/output/dialog rails with flow names)

### Debugging Capabilities

Use the detailed logging options to understand what's happening in your guardrails:

```python
logging_options = {
    "output_vars": True,
    "log": {
        "activated_rails": True, 
        "llm_calls": True,
        "internal_events": True
    }
}

result = rails.generate(messages=messages, options=logging_options)
```

## Current Limitations and Workarounds

### RunnableRails Integration

The `RunnableRails` wrapper has significant limitations with action registration. Unlike `LLMRails`, there's no direct way to register custom actions with a `RunnableRails` instance, which leads to "Action not found" errors when using custom actions in your Colang flows.

#### Workaround Solution

Instead of using RunnableRails, you can create a function that uses LLMRails directly but formats the input/output to be compatible with LangChain:

```python
def get_guardrailed_response(topic):
    """Get guardrailed response using LLMRails directly."""
    # Create a fresh rails instance
    rails = LLMRails(custom_config)
    # Register the actions
    rails.register_action(check_input_toxicity)
    rails.register_action(check_output_toxicity)
    
    messages = [
        {"role": "user", "content": f"Tell me a joke about {topic}"}
    ]
    try:
        result = rails.generate(messages=messages)
        # Extract content from response
        if isinstance(result, dict) and 'content' in result:
            return result['content']
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"
```

This approach gives you the full power of guardrails with custom actions while still allowing integration with your LangChain application.

## Comparing Example Approaches

The examples in this repository demonstrate different guardrail implementation patterns, each with unique strengths:

| Example | Primary Focus | Key Guardrail Features | Best For |
|---------|---------------|------------------------|----------|
| `simple_main.py` | Basic integration | Toxicity checks, minimal config | Getting started with guardrails |
| `main.py` | Multiple techniques | Key mapping, streaming, various integrations | Advanced integration patterns |
| `rag_example.py` | Hallucination prevention | Context verification, selective application | RAG systems and knowledge-based applications |
| `multi_model_example.py` | Purpose-specific guardrails | Content moderation, transformation of problematic content | Creative content generation, multiple use cases |
| `key_mapping_example.py` | Custom I/O formats | Format adaptation, metadata preservation | API integration, custom schemas |

### Example Usage Scenarios

- **Content Creation with Safety**: Use the multi-model example approach when generating creative content that requires different levels of moderation based on audience or topic
- **Knowledge Systems and RAG**: Use the RAG example approach when building systems that need to avoid hallucinations and maintain factual accuracy
- **Mixed-Purpose Applications**: Combine techniques from different examples when building applications that serve multiple purposes (e.g., a system that provides both creative content and factual responses)
- **API Integration**: Use the key mapping approach when integrating with existing APIs that have specific input/output format requirements

### Output Comparison

The examples demonstrate how guardrails transform outputs in different scenarios:

1. **Factual Queries (RAG Example)**:
   - Without guardrails: May generate plausible but unsupported claims
   - With guardrails: Acknowledges limitations of available information

2. **Creative Content (Multi-Model Example)**:
   - Without guardrails: May include inappropriate themes, violence, or disturbing content
   - With guardrails: Transforms problematic content into family-friendly alternatives

3. **General Conversation (Simple Example)**:
   - Without guardrails: May respond to harmful or toxic inputs
   - With guardrails: Refuses to engage with harmful content

This side-by-side comparison clearly illustrates the value of guardrails in creating safer, more reliable AI applications.

## Documentation References

For more details, see the official NeMo Guardrails documentation:

- [RunnableRails Integration](https://docs.nvidia.com/nemo/guardrails/latest/user-guides/langchain/runnable-rails.html)
- [Detailed Logging](https://docs.nvidia.com/nemo/guardrails/latest/user-guides/detailed-logging/README.html) 