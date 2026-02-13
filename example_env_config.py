"""
Example 1: Using Environment Variables (Recommended)

This example demonstrates how to configure crewAI to use a custom OpenAI-compatible API
by setting environment variables in a .env file.

Prerequisites:
1. Create a .env file in the project root with:
   OPENAI_API_KEY=your-api-key
   OPENAI_BASE_URL=https://code.newcli.com/codex/v1
   OPENAI_MODEL_NAME=gpt-5.2
"""

from crewai import Agent, Crew, Task


# Create an agent - it will automatically use the environment variables
agent = Agent(
    role="Research Assistant",
    goal="Provide accurate and helpful information",
    backstory="You are a knowledgeable assistant with expertise in various topics.",
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
    print("Running crewAI with custom API configuration from environment variables...")
    result = crew.kickoff()
    print("\n" + "="*50)
    print("Result:")
    print("="*50)
    print(result)
