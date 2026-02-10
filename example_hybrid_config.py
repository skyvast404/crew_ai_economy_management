"""
Example 3: Hybrid Configuration

This example demonstrates how to configure crewAI using a combination of
environment variables and code configuration.
"""

import os
from crewai import Agent, Task, Crew, LLM

# Set environment variables programmatically
os.environ["OPENAI_API_KEY"] = "sk-ant-oat01-G2Hn_a2kkZ_Z-KdrawsLVOssknJxBNo2X6XCLteEQEmNqeeGH-6LgyGV-WQFw8_LKvQp6vT-2XnbheJUazTzyba4aRO7xAA"
os.environ["OPENAI_BASE_URL"] = "https://code.newcli.com/codex/v1"

# Create LLM with model name and optional parameters
# API key and base_url are read from environment variables
llm = LLM(
    model="gpt-5.2",
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
    print("Running crewAI with hybrid configuration...")
    result = crew.kickoff()
    print("\n" + "="*50)
    print("Result:")
    print("="*50)
    print(result)
