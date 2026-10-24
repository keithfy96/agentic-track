import asyncio

asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import gradio as gr
from langgraph.prebuilt import ToolNode, tools_condition
import requests
import os

from langchain.agents import Tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

load_dotenv(override=True)


# ============================================================
# STATE
# ============================================================


class State(TypedDict):
    messages: Annotated[list, add_messages]


# ============================================================
# PUSH TOOL
# ============================================================

pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_user = os.getenv("PUSHOVER_USER")
pushover_url = "https://api.pushover.net/1/messages.json"


def push(text: str):
    """Send a push notification to the user via Pushover."""
    if not pushover_token or not pushover_user:
        return "Pushover credentials not configured."
    try:
        resp = requests.post(
            pushover_url,
            data={"token": pushover_token, "user": pushover_user, "message": text},
            timeout=10,
        )
        if resp.status_code == 200:
            return "Push notification sent."
        return f"Failed to send push: {resp.status_code} {resp.text}"
    except Exception as e:
        return f"Error sending push: {e}"


push_tool = Tool(
    name="send_push_notification",
    func=push,
    description="Send a push notification through Pushover.",
)


# ============================================================
# PLAYWRIGHT TOOLS (LAZY + ASYNC)
# ============================================================

# We load the Playwright toolkit only when a tool is actually called.
# No browser is created at import time.


async def _load_playwright_tools():
    from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
    from langchain_community.tools.playwright.utils import (
        create_async_playwright_browser,
    )

    async_browser = await create_async_playwright_browser(headless=False)
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
    tools = toolkit.get_tools()
    return {t.name: t for t in tools}


async def _call_playwright_tool(tool_name: str, **kwargs):
    tools = await _load_playwright_tools()
    tool = tools.get(tool_name)
    if tool is None:
        raise ValueError(f"Playwright tool {tool_name} not found.")

    # Prefer async if available, otherwise run sync in a thread
    if hasattr(tool, "ainvoke"):
        return await tool.ainvoke(kwargs)
    else:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, tool.invoke, kwargs)


# Actual async tool functions wired into LangChain Tool.coroutine


async def navigate_browser_tool(url: str):
    """Navigate the Playwright browser to a URL."""
    return await _call_playwright_tool("navigate_browser", url=url)


async def extract_text_tool():
    """Extract visible text from the current webpage."""
    return await _call_playwright_tool("extract_text")


navigate_tool = Tool(
    name="navigate_browser",
    description="Navigate to a webpage using Playwright.",
    func=lambda url: "This tool must be used asynchronously.",
    coroutine=navigate_browser_tool,
)

extract_text_tool_def = Tool(
    name="extract_text",
    description="Extract visible text from the current page using Playwright.",
    func=lambda: "This tool must be used asynchronously.",
    coroutine=extract_text_tool,
)


# ============================================================
# ALL TOOLS
# ============================================================

all_tools = [navigate_tool, extract_text_tool_def, push_tool]


# ============================================================
# LLM + GRAPH
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(all_tools)


def chatbot(state: State):
    """LLM node that may choose to call tools."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=all_tools))
graph_builder.add_conditional_edges("chatbot", tools_condition, "tools")
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


# ============================================================
# GRADIO FRONTEND
# ============================================================

config = {"configurable": {"thread_id": "1"}}


async def chat(user_input: str, history):
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
    )
    return result["messages"][-1].content


gr.ChatInterface(chat, type="messages").launch()
