import asyncio
from rich.console import Console
from rich.markdown import Markdown
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from dataclasses import dataclass
from langchain.agents import AgentState, create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage
from langchain.agents.middleware import wrap_model_call, dynamic_prompt, HumanInTheLoopMiddleware
from langchain.agents.middleware import ModelRequest, ModelResponse
from typing import Callable

load_dotenv()

@dataclass
class EmailContext:
    email_address: str = "julie@example.com"
    password: str = "password123"

class AuthenticatedState(AgentState):
    authenticated: bool

@tool
def check_inbox() -> str:
    """Check the inbox for recent emails"""
    return """
    Hi Julie, 
    I'm going to be in town next week and was wondering if we could grab a coffee?
    - best, Jane (jane@example.com)
    """

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an response email"""
    return f"Email sent to {to} with subject {subject} and body {body}"

@tool
def authenticate(email: str, password: str, runtime: ToolRuntime) -> Command:
    """Authenticate the user with the given email and password"""
    if email == runtime.context.email_address and password == runtime.context.password:
        return Command(
            update={
                "authenticated": True,
                "messages": [
                    ToolMessage("Successfully authenticated", tool_call_id=runtime.tool_call_id)
                ],
            }
        )
    else:
        return Command(
            update={
                "authenticated": False,
                "messages": [
                    ToolMessage("Authentication failed", tool_call_id=runtime.tool_call_id)
                ],
            }
        )

@wrap_model_call
async def dynamic_tool_call(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    authenticated = request.state.get("authenticated")
    if authenticated:
        tools = [check_inbox, send_email]
    else:
        tools = [authenticate]
    request = request.override(tools=tools)
    return await handler(request)

authenticated_prompt = "You are a helpful assistant that can check the inbox and send emails."
unauthenticated_prompt = "You are a helpful assistant that can authenticate users."

@dynamic_prompt
def dynamic_prompt_func(request: ModelRequest) -> str:
    authenticated = request.state.get("authenticated")
    return authenticated_prompt if authenticated else unauthenticated_prompt

model = ChatOllama(model="qwen3:14b", temperature=0)

agent = create_agent(
        model=model,
        tools=[authenticate, check_inbox, send_email],
        state_schema=AuthenticatedState,
        context_schema=EmailContext,
        middleware=[
            dynamic_tool_call,
            dynamic_prompt_func,
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "authenticate": False,
                    "check_inbox": False,
                    "send_email": True, # This will trigger an interrupt
                }
            ),
        ],
    )

async def run_email_agent():
    console = Console()
    
    # 1. Initialize context as a dictionary for Pydantic
    context = EmailContext().__dict__ 

    print("🔐 Starting Secure Email Agent...")

    # 2. Step One: Authentication
    query_1 = "Please authenticate me. My email is julie@example.com and password is password123"
    print(f"\n[User]: {query_1}")
    
    # Corrected variable 'state' to 'initial_state'
    initial_state = {"messages": [("user", query_1)], "authenticated": False}
    
    response_1 = await agent.ainvoke(initial_state, context=context)
    
    # The LLM might need a second pass to acknowledge the tool result
    if response_1["messages"][-1].type == "tool":
         response_1 = await agent.ainvoke(response_1, context=context)

    auth_content = response_1["messages"][-1].content
    console.print(Markdown(f"**Agent:** {auth_content}"))

    # 3. Step Two: Check Inbox
    if response_1.get("authenticated"):
        query_2 = "Great. Now check my inbox and tell me if I have any mail."
        print(f"\n[User]: {query_2}")
        
        # Append the new message to the existing history
        state_2 = {**response_1, "messages": response_1["messages"] + [("user", query_2)]}
        response_2 = await agent.ainvoke(state_2, context=context)
        
        # Again, if the last message is just the tool output, invoke again to get the text response
        if response_2["messages"][-1].type == "tool":
            response_2 = await agent.ainvoke(response_2, context=context)

        inbox_content = response_2["messages"][-1].content
        console.print(Markdown(f"**Agent:** {inbox_content}"))

        # 4. Step Three: Send Email (HITL)
        query_3 = "Reply to Jane and tell her I'd love to grab coffee on Wednesday."
        print(f"\n[User]: {query_3}")
        
        state_3 = {**response_2, "messages": response_2["messages"] + [("user", query_3)]}
        response_3 = await agent.ainvoke(state_3, context=context)
        
        # Handle the HITL interruption
        # If it's interrupted, you would usually prompt the user for 'yes/no' 
        # For this execution, we'll just print the current status
        final_content = response_3["messages"][-1].content
        console.print(Markdown(f"**Agent Status:** {final_content}"))
    else:
        print("❌ Authentication failed.")

if __name__ == "__main__":
    asyncio.run(run_email_agent())