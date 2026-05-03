
"""
Research Agent Core Implementation.

This module implements a research agent that can perform iterative web searches
and synthesis to answer complex research questions.
"""

from typing import Literal

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, filter_messages
from langchain_anthropic import ChatAnthropic

# custom imports
from src.agent_prompts import research_agent_prompt, compress_research_system_prompt, compress_research_human_message
from src.agent_tools import tavily_search, think_tool
from src.state_research import ResearcherState, ResearcherOutputState
from src.utils import date_today_str


############ INITIALIZATION SETUP ##########

tools = [tavily_search, think_tool]
tools_by_name = {tool.name: tool for tool in tools}

model = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
model_with_tools = model.bind_tools(tools)
context_compression_model = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=32000, temperature=0)

MAX_TOOL_ITERATIONS = 3


########### RESEARCH AGENT NODES ##########

def llm_call(state: ResearcherState):
    """
    Analyze current state and decide on next actions.

    The model analyzes the current conversation state and decides whether to:
    1. Call search tools to gather more information
    2. Provide a final answer based on gathered information

    When the tool-call iteration cap is reached, the unbound model is invoked
    with a budget-exhausted nudge so it emits a final summary instead of more
    tool_use blocks. This keeps Anthropic's tool_use / tool_result pairing
    invariant intact when control falls through to compress_research.

    Returns updated state with the model's response.
    """
    at_cap = state.get("tool_call_iterations", 0) >= MAX_TOOL_ITERATIONS
    research_agent_prompt_formatted = research_agent_prompt.format(date=date_today_str())
    if at_cap:
        research_agent_prompt_formatted += (
            "\n\nTool budget exhausted. Synthesize findings from prior tool results now. "
            "Do not call any more tools."
        )
    chosen_model = model if at_cap else model_with_tools
    return {
        "researcher_messages": [
            chosen_model.invoke(
                [SystemMessage(content=research_agent_prompt_formatted)] + state["researcher_messages"]
            )
        ]
    }

def tool_node(state: ResearcherState):
    """
    Execute all tool calls from the previous LLM response.
    
    Executes all tool calls from the previous LLM responses.
    Returns updated state with tool execution results.
    """
    tool_calls = state["researcher_messages"][-1].tool_calls

    # Always emit one ToolMessage per tool_call so Anthropic's
    # `tool_use` / `tool_result` pairing invariant is preserved even when
    # a tool raises or the model emits an unknown tool name. Otherwise the
    # next llm_call invocation fails with a 400 from Anthropic.
    # Cap each tool observation so accumulated researcher_messages cannot
    # exceed the 200k context window across MAX_TOOL_ITERATIONS turns.
    MAX_OBS_CHARS = 30000

    tool_outputs = []
    for tool_call in tool_calls:
        name = tool_call.get("name", "")
        try:
            tool = tools_by_name[name]
            observation = tool.invoke(tool_call["args"])
        except Exception as e:
            observation = f"Tool '{name}' failed: {e}"
        content = str(observation)
        if len(content) > MAX_OBS_CHARS:
            content = content[:MAX_OBS_CHARS] + "\n...[truncated]"
        tool_outputs.append(
            ToolMessage(
                content=content,
                name=name or "unknown",
                tool_call_id=tool_call["id"],
            )
        )

    return {
        "researcher_messages": tool_outputs,
        "tool_call_iterations": state.get("tool_call_iterations", 0) + 1,
    }

def compress_research(state: ResearcherState) -> dict:
    """Compress research findings into a concise summary.

    Takes all the research messages and tool outputs and creates
    a compressed summary suitable for the supervisor's decision-making.
    """

    system_message = compress_research_system_prompt.format(date=date_today_str())
    human_message = compress_research_human_message.format(
        research_topic=state.get("research_topic", "")
    )
    messages = [SystemMessage(content=system_message)] + state.get("researcher_messages", []) + [HumanMessage(content=human_message)]
    response = context_compression_model.invoke(messages)

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

# Conditional function to determine whether to continue research or provide final answer
def should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    """
    Determine whether to continue research or provide final answer.

    The iteration cap is enforced inside llm_call (it switches to the unbound
    model at the cap), so by the time control reaches here the last message
    either has tool_calls (continue) or is a final synthesis (compress).

    Returns:
        "tool_node": Continue to tool execution
        "compress_research": Stop and compress research
    """
    last_message = state["researcher_messages"][-1]
    return "tool_node" if last_message.tool_calls else "compress_research"


########### RESEARCH AGENT GRAPH ##########

research_agent_graph = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

# Nodes
research_agent_graph.add_node("llm_call", llm_call)
research_agent_graph.add_node("tool_node", tool_node)
research_agent_graph.add_node("compress_research", compress_research)

# Edges
research_agent_graph.add_edge(START, "llm_call")
research_agent_graph.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node", # Continue research loop
        "compress_research": "compress_research", # Provide final answer
    },
)
research_agent_graph.add_edge("tool_node", "llm_call")
research_agent_graph.add_edge("compress_research", END)

researcher_agent = research_agent_graph.compile()
