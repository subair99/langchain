import asyncio
from typing import Callable
from dataclasses import dataclass
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain.agents import AgentState, create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage
from langchain.agents.middleware import (
    wrap_model_call, 
    dynamic_prompt, 
    HumanInTheLoopMiddleware,
    ModelRequest, 
    ModelResponse
)

load_dotenv()

# --- 1. SCHEMAS (Refactored to Pydantic to stop warnings) ---

class EmailContext(BaseModel):
    email_address: str = "julie@example.com"
    password: str = "password123"

class AuthenticatedState(AgentState):
    authenticated: bool = False

# --- 2. TOOLS ---

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
    return Command(
        update={
            "authenticated": False,
            "messages": [
                ToolMessage("Authentication failed", tool_call_id=runtime.tool_call_id)
            ],
        }
    )

# --- 3. MIDDLEWARE ---

@wrap_model_call
async def dynamic_tool_call(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    authenticated = request.state.get("authenticated", False)
    tools = [check_inbox, send_email] if authenticated else [authenticate]
    request = request.override(tools=tools)
    return await handler(request)

@dynamic_prompt
def dynamic_prompt_func(request: ModelRequest) -> str:
    authenticated = request.state.get("authenticated", False)
    return "You are a helpful assistant for emails." if authenticated else "You are an authentication assistant."

# --- 4. AGENT INITIALIZATION ---

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
            interrupt_on={"send_email": True}
        ),
    ],
)

# --- 5. EXECUTION BLOCK ---

async def run_email_agent():
    console = Console()
    context = EmailContext() # Pydantic object works directly now
    state = {"messages": [], "authenticated": False}

    print("🔐 Starting Secure Email Agent...")

    # FLOW: Authenticate -> Check -> Send (with Approval)
    steps = [
        "Please authenticate me. My email is julie@example.com and password is password123",
        "Great. Now check my inbox and tell me if I have any mail.",
        "Reply to Jane and tell her I'd love to grab coffee on Wednesday."
    ]

    for query in steps:
        print(f"\n[User]: {query}")
        state["messages"].append(("user", query))
        
        # Invoke Agent
        response = await agent.ainvoke(state, context=context)
        
        # Check for HITL Interrupt (Agent wants to call send_email)
        last_msg = response["messages"][-1]
        
        if hasattr(last_msg, "tool_calls") and any(tc['name'] == 'send_email' for tc in last_msg.tool_calls):
            console.print("\n[bold yellow]⚠️  HITL INTERRUPT: The agent wants to send an email.[/bold yellow]")
            confirm = input("Do you approve this action? (yes/no): ")
            
            if confirm.lower() == 'yes':
                # Resume by invoking again with the state that contained the tool call
                response = await agent.ainvoke(response, context=context)
            else:
                print("❌ Action cancelled.")
                break

        # Follow-up: If the last message is just a tool result, let the LLM talk back to the user
        if response["messages"][-1].type == "tool":
            response = await agent.ainvoke(response, context=context)

        # Update loop state
        state = response
        
        # Display output
        console.print(Markdown(f"**Agent:** {state['messages'][-1].content}"))

if __name__ == "__main__":
    asyncio.run(run_email_agent())