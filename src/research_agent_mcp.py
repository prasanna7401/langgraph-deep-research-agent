
"""
Research Agent with MCP Integration.

This module implements a research agent that integrates with Model Context Protocol (MCP)
servers to access tools and resources. The agent demonstrates how to use MCP filesystem
server for local document research and analysis.

Key features:
- MCP server integration for tool access
- Async operations for concurrent tool execution (required by MCP protocol)
- Filesystem operations for local document research
- Secure directory access with permission checking
- Research compression for efficient processing
- Lazy MCP client initialization for LangGraph Platform compatibility
"""

from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, filter_messages
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END

from src.agent_prompts import research_agent_prompt_with_mcp, compress_research_system_prompt, compress_research_human_message
from src.state_research import ResearcherState, ResearcherOutputState
from src.agent_tools import think_tool
from src.utils import date_today_str, current_dir

# ===== CONFIGURATION =====

# MCP server configuration for filesystem access
mcp_config = {
    "filesystem": {
        "command": "npx",
        "args": [
            "-y",  # Auto-install if needed
            "@modelcontextprotocol/server-filesystem",
            str(current_dir() / "files")  # Path to research documents
        ],
        "transport": "stdio"  # Communication via stdin/stdout
    }
}

# Global client variable - will be initialized lazily
_client = None
_mcp_tools_cache = None

def get_mcp_client():
    """Get or initialize MCP client lazily to avoid issues with LangGraph Platform."""
    global _client
    if _client is None:
        _client = MultiServerMCPClient(mcp_config)
    return _client

async def _get_mcp_tools():
    """Fetch MCP tool list once and cache it. Tool list is static for the session."""
    global _mcp_tools_cache
    if _mcp_tools_cache is None:
        client = get_mcp_client()
        _mcp_tools_cache = await client.get_tools()
    return _mcp_tools_cache

# Initialize models - change models as per your requirement
compress_model = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=32000, temperature=0)
model = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

MAX_TOOL_ITERATIONS = 5

# ===== AGENT NODES =====

async def llm_call(state: ResearcherState): # notice the async keyword here, it's required for MCP tool calls
    """Analyze current state and decide on tool usage with MCP integration.

    This node:
    1. Retrieves available tools from MCP server (cached after first fetch)
    2. Binds tools to the language model
    3. Processes user input and decides on tool usage asynchronously

    Returns updated state with model response.
    """
    mcp_tools = await _get_mcp_tools()
    tools = mcp_tools + [think_tool]

    model_with_tools = model.bind_tools(tools)

    researcher_system_message = research_agent_prompt_with_mcp.format(date=date_today_str())

    response = await model_with_tools.ainvoke(
        [SystemMessage(content=researcher_system_message)] + state["researcher_messages"]
    )

    return {"researcher_messages": [response]}

async def tool_node(state: ResearcherState):
    """Execute tool calls using MCP tools.

    This node:
    1. Retrieves current tool calls from the last message
    2. Executes all tool calls using async operations (required for MCP)
    3. Returns formatted tool results

    Note: MCP requires async operations due to inter-process communication
    with the MCP server subprocess. This is unavoidable.
    """
    tool_calls = state["researcher_messages"][-1].tool_calls

    mcp_tools = await _get_mcp_tools()
    tools = mcp_tools + [think_tool]
    tools_by_name = {tool.name: tool for tool in tools}

    observations = []
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        if tool_call["name"] == "think_tool":
            observation = tool.invoke(tool_call["args"])
        else:
            observation = await tool.ainvoke(tool_call["args"])
        observations.append(observation)

    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"],
        )
        for observation, tool_call in zip(observations, tool_calls)
    ]

    return {
        "researcher_messages": tool_outputs,
        "tool_call_iterations": state.get("tool_call_iterations", 0) + 1,
    }

def compress_research(state: ResearcherState) -> dict:
    """Compress research findings into a concise summary.

    Takes all the research messages and tool outputs and creates
    a compressed summary suitable for further processing or reporting.

    This function filters out think_tool calls and focuses on substantive
    file-based research content from MCP tools.
    """

    compress_research_system_message = compress_research_system_prompt.format(date=date_today_str())
    compress_research_human = compress_research_human_message.format(
        research_topic=state.get("research_topic", "")
    )
    messages = [
        SystemMessage(content=compress_research_system_message)] + state.get("researcher_messages", []) + [HumanMessage(content=compress_research_human)]

    response = compress_model.invoke(messages)

    # Extract raw notes from tool and AI messages
    raw_notes = [
        str(m.content) for m in filter_messages(
            state["researcher_messages"], 
            include_types=["tool", "ai"]
        )
    ]

    return {
        "compressed_research": str(response.content),
        "raw_notes": ["\n".join(raw_notes)]
    }

# ===== ROUTING LOGIC =====

def should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    """Determine whether to continue with tool execution or compress research.

    Determines whether to continue with tool execution or compress research
    based on whether the LLM made tool calls.
    """
    last_message = state["researcher_messages"][-1]

    if not last_message.tool_calls:
        return "compress_research"

    if state.get("tool_call_iterations", 0) >= MAX_TOOL_ITERATIONS:
        return "compress_research"

    return "tool_node"

# ===== GRAPH CONSTRUCTION =====

# Build the agent workflow
agent_builder_mcp = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

# Add nodes to the graph
agent_builder_mcp.add_node("llm_call", llm_call)
agent_builder_mcp.add_node("tool_node", tool_node)
agent_builder_mcp.add_node("compress_research", compress_research)

# Add edges to connect nodes
agent_builder_mcp.add_edge(START, "llm_call")
agent_builder_mcp.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node",        # Continue to tool execution
        "compress_research": "compress_research",  # Compress research findings
    },
)
agent_builder_mcp.add_edge("tool_node", "llm_call")  # Loop back for more processing
agent_builder_mcp.add_edge("compress_research", END)

# Compile the agent
agent_mcp = agent_builder_mcp.compile()
