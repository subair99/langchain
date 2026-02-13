from dotenv import load_dotenv

load_dotenv()

from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:

    """Search the web for information"""

    return tavily_client.search(query)

system_prompt = """

You are a personal chef. The user will give you a list of ingredients they have left over in their house.

Using the web search tool, search the web for recipes that can be made with the ingredients they have.

Return recipe suggestions and eventually the recipe instructions to the user, if requested.

"""

from langchain_groq import ChatGroq

model = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)


from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=[web_search],
    system_prompt=system_prompt
)


# 1. Prepare the input
user_input = {"messages": [("user", "I have leftover chicken and white rice. Find me a recipe!")]}

# 2. Step on the gas (Invoke the agent)
print("Chef is thinking and searching...")
response = agent.invoke(user_input)

# 3. Print the final message from the AI
# In the modern LangGraph/LangChain format, the last message is the answer.
print("\n--- CHEF'S SUGGESTION ---")
print(response["messages"][-1].content)