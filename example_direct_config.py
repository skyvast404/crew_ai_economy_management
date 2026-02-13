"""
Example 2: Direct Code Configuration

This example demonstrates how to configure crewAI to use a custom OpenAI-compatible API
by directly specifying the LLM configuration in code.
"""

import os

from crewai import LLM, Agent, Crew, Task


# Config loaded from .env via load_dotenv()
# OPENAI_API_KEY, OPENAI_BASE_URL, and OPENAI_MODEL_NAME are read from environment
llm = LLM(
    model=os.environ.get("OPENAI_MODEL_NAME", "gemini-3-pro-low"),
    temperature=0.7,
    max_tokens=2000
)

# Create an agent with the custom LLM
agent = Agent(
    role="Research Assistant",
    goal="Provide accurate and helpful information",
    backstory="You are a knowledgeable assistant with expertise in various topics.",
    llm=llm,
    verbose=True
)

# Create a task
task = Task(
    description="Explain what crewAI is and its main features in 3 sentences.",
    expected_output="A concise explanation of crewAI with its key features.",
    agent=agent
)

# Create and run the crew
crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True
)

if __name__ == "__main__":
    print("Running crewAI with custom API configuration from code...")
    result = crew.kickoff()
    print("\n" + "="*50)
    print("Result:")
    print("="*50)
    print(result)
