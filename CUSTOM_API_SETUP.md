# Custom OpenAI-Compatible API Configuration for crewAI 1.9.3

This guide explains how to configure crewAI to use a custom OpenAI-compatible API endpoint.

## Your Configuration

- **Base URL**: `https://code.newcli.com/codex/v1`
- **Model**: `gpt-5.2`
- **API Key**: Configured in `.env` file

## Configuration Methods

### Method 1: Environment Variables (Recommended)

The simplest approach is to use a `.env` file in your project root.

**Step 1**: Create a `.env` file with the following content:

```bash
OPENAI_API_KEY=sk-ant-oat01-G2Hn_a2kkZ_Z-KdrawsLVOssknJxBNo2X6XCLteEQEmNqeeGH-6LgyGV-WQFw8_LKvQp6vT-2XnbheJUazTzyba4aRO7xAA
OPENAI_BASE_URL=https://code.newcli.com/codex/v1
OPENAI_MODEL_NAME=gpt-5.2
CREWAI_DISABLE_TELEMETRY=true
```

**Step 2**: Use in your code (see `example_env_config.py`):

```python
from crewai import Agent, Task, Crew

agent = Agent(
    role="Research Assistant",
    goal="Provide information",
    backstory="You are helpful"
)
```

The configuration is automatically loaded from environment variables.

### Method 2: Direct Code Configuration

Configure the LLM directly in your code (see `example_direct_config.py`):

```python
from crewai import Agent, LLM

llm = LLM(
    model="gpt-5.2",
    base_url="https://code.newcli.com/codex/v1",
    api_key="your-api-key",
    temperature=0.7,
    max_tokens=2000
)

agent = Agent(
    role="Research Assistant",
    goal="Provide information",
    backstory="You are helpful",
    llm=llm
)
```

### Method 3: Hybrid Configuration

Combine environment variables with code configuration (see `example_hybrid_config.py`):

```python
import os
from crewai import LLM

os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["OPENAI_BASE_URL"] = "https://code.newcli.com/codex/v1"

llm = LLM(model="gpt-5.2", temperature=0.7)
```

## Key Configuration Parameters

### Required Parameters

- `model`: Model name (e.g., "gpt-5.2")
- `base_url`: API endpoint URL
- `api_key`: Your API authentication key

### Optional Parameters

- `temperature`: 0-2 (default: 0.7) - Controls randomness
- `max_tokens`: Maximum output tokens
- `top_p`: Nucleus sampling parameter
- `timeout`: Request timeout in seconds

## Quick Start

1. Run any of the example files:

```bash
python example_env_config.py
```

