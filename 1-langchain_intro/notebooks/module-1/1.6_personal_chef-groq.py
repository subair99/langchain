import os
from dotenv import load_dotenv
from typing import Dict, Any
from tavily import TavilyClient
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

load_dotenv()

# 1. Setup Search Tool
tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information"""
    return tavily_client.search(query)

# 2. Setup LLM (Using the 120B model on Groq)
model = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)

# 3. Define the Persona
system_prompt = """
You are a personal chef. The user will give you a list of ingredients.
Using the web search tool, find recipes and return instructions.
"""

# 4. Initialize the Agent
agent = create_react_agent(
    model=model,
    tools=[web_search],
    state_modifier=system_prompt
)

# 5. EXECUTION BLOCK (This generates the output)
if __name__ == "__main__":
    print("👨‍🍳 Chef is standing by...")
    
    # Define your request
    inputs = {"messages": [HumanMessage(content="I have leftover chicken and rice. What can I make?")]}
    
    # Run the agent and capture the result
    result = agent.invoke(inputs)
    
    # Extract and print the final response
    final_answer = result["messages"][-1].content
    print("\n--- CHEF'S RECOMMENDATION ---")
    print(final_answer)