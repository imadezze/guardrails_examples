# Common Files for NeMo Guardrails Examples

This directory contains shared resources used across multiple examples in the NeMo Guardrails with LangChain integration project.

## Directory Structure

- **`config/`**: Contains shared guardrails configuration
  - **`config.yml`**: Base configuration for NeMo Guardrails
  - **`colang_files/`**: Colang logic for guardrails
    - **`toxicity.co`**: Common toxicity check flows
  - **`rails/`**: Additional guardrail definitions
  
- **`utils/`**: Utility functions and helpers
  - **`utils.py`**: Common utility functions for setting up examples

- **`data/`**: Shared data resources for examples

## Usage

The files in this directory are imported and used by the individual examples. This shared approach ensures consistency across examples and reduces code duplication.

### Using the Configuration

```python
from pathlib import Path
from nemoguardrails import RailsConfig

# Load the guardrails configuration
config_path = Path(__file__).parent.parent / "common" / "config"
config = RailsConfig.from_path(config_path)
```

### Using Utility Functions

```python
from examples.common.utils import create_example_config, check_environment

# Check if environment is properly set up
if not check_environment():
    # Handle missing environment variables
    pass

# Create a config with custom colang files
example_dir = Path(__file__).parent
create_example_config(example_dir, {
    "custom.co": "define flow custom flow\n  ..."
})
```

## Extending

When adding new examples, you can extend the common resources:

1. Add new utility functions to `utils.py`
2. Add shared data to `data/`
3. Add common guardrail definitions to the appropriate files in `config/` 