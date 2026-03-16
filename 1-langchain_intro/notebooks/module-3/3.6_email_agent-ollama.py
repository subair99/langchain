import asyncio
import warnings
from typing import Callable, TypedDict
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

# 0. Suppress persistent Pydantic serialization warnings globally
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

load_dotenv()

# --- 1. SCHEMAS ---

# Using TypedDict resolves the "PydanticSerializationUnexpectedValue" warning
class EmailContext(TypedDict):
    email_address: str
    password: str

class AuthenticatedState(AgentState):
    authenticated: bool

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
    if email == runtime.context["email_address"] and password == runtime.context["password"]:
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

# --- 5. EXECUTION BLOCK (Updated) ---

async def run_email_agent():
    console = Console()
    context: EmailContext = {
        "email_address": "julie@example.com",
        "password": "password123"
    }
    state = {"messages": [], "authenticated": False}

    print("🔐 Starting Secure Email Agent (Type 'exit' to quit)...")

    while True:
        query = input("\n[User]: ")
        if query.lower() == "exit":
            break
            
        state["messages"].append(("user", query))
        
        # 1. Invoke Agent
        response = await agent.ainvoke(state, context=context)
        
        # 2. Check for HITL Interrupt
        last_msg = response["messages"][-1]
        if hasattr(last_msg, "tool_calls") and any(tc['name'] == 'send_email' for tc in last_msg.tool_calls):
            console.print("\n[bold yellow]⚠️  HITL INTERRUPT: The agent wants to send an email.[/bold yellow]")
            tc = next(tc for tc in last_msg.tool_calls if tc['name'] == 'send_email')
            print(f"To: {tc['args'].get('to')}\nBody: {tc['args'].get('body')}")
            
            confirm = input("\nDo you approve this action? (yes/no): ")
            if confirm.lower() == 'yes':
                response = await agent.ainvoke(response, context=context)
            else:
                print("❌ Action cancelled.")
                # We do NOT break here; we just continue the conversation
                continue

        # 3. Follow-up: Ensure the agent provides a text response
        while response["messages"][-1].type == "tool":
            response = await agent.ainvoke(response, context=context)

        state = response
        
        # 4. Display AI response
        ai_responses = [m for m in state["messages"] if hasattr(m, 'type') and m.type == 'ai' and m.content]
        if ai_responses:
            console.print(Markdown(f"**Agent:** {ai_responses[-1].content}"))
            

if __name__ == "__main__":
    asyncio.run(run_email_agent())