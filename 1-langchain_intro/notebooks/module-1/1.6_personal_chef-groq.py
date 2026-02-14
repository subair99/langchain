import os
from dotenv import load_dotenv
from typing import Dict, Any
from tavily import TavilyClient
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent

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

# 4. Initialize the Agent (Using your correct syntax)
agent = create_agent(
    model=model,
    tools=[web_search],
    system_prompt=system_prompt
)

# 5. EXECUTION BLOCK
if __name__ == "__main__":
    print("👨‍🍳 Chef is checking the pantry...")
    
    # Send a prompt to the agent
    query = "I have leftover chicken and rice. Give me a creative recipe!"
    
    # Use invoke to get the response
    response = agent.invoke({"messages": [("user", query)]})
    
    # Print the output
    print("\n--- CHEF'S SUGGESTION ---")
    # Note: Accessing the content from the last message in the returned list
    print(response["messages"][-1].content)