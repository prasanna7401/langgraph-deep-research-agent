
"""
Multi-agent supervisor for coordinating research across multiple specialized agents.

This module implements a supervisor pattern where:
1. A supervisor agent coordinates research activities and delegates tasks
2. Multiple researcher agents work on specific sub-topics independently
3. Results are aggregated and compressed for final reporting

The supervisor uses parallel research execution to improve efficiency while
maintaining isolated context windows for each research topic.
"""
import asyncio

from typing import Literal
from langchain_core.messages import (
    HumanMessage, 
    BaseMessage, 
    SystemMessage, 
    ToolMessage,
    filter_messages
)
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_anthropic import ChatAnthropic

# custom imports
from src.agent_prompts import lead_researcher_prompt
from src.research_agent import researcher_agent
from src.state_multi_agent_supervisor import SupervisorState, ConductResearch, ResearchComplete
from src.agent_tools import think_tool
from src.utils import date_today_str


def get_notes_from_tool_calls(messages: list[BaseMessage]) -> list[str]:
    """
    Extract notes from ToolMessage instances in the message history.

    This function retrieves the compressed research findings that sub-agents
    return as ToolMessage content. When the supervisor delegates research to
    sub-agents via ConductResearch tool calls, each sub-agent returns its
    compressed findings as the content of a ToolMessage. This function
    extracts all such ToolMessage content to compile the final research notes.
    
    Args:
        messages: List of messages from supervisor's conversation history
        
    Returns:
        List of research note strings extracted from ToolMessage objects
    """
    tool_messages = filter_messages(messages, include_types="tool")
    
    notes = [msg.content for msg in tool_messages if isinstance(msg.content, str)]
    
    return notes

# Ensure async compatibility for Jupyter environments
try:
    import nest_asyncio
    # Only apply if running in Jupyter/IPython environment
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            nest_asyncio.apply()
    except ImportError:
        pass  # Not in Jupyter, no need for nest_asyncio
except ImportError:
    pass  # nest_asyncio not available, proceed without it

########### INITIALIZATION ##########

model = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
supervisor_tools = [ConductResearch, ResearchComplete, think_tool]
supervisor_model = model.bind_tools(supervisor_tools)

max_research_iterations = 3 # maximum number of research iterations before supervisor compiles final report
max_concurrent_researchers = 2 # maximum number of concurrent researcher agents to run in parallel for efficiency

########### SUPERVISOR AGENT IMPLEMENTATION ##########

# Nodes

async def supervisor(state: SupervisorState) -> Command[Literal["supervisor_tools"]]:
    """
    Coordinate research activities.
    
    Analyzes the research brief and current progress to decide:
    - What research topics need investigation
    - Whether to conduct parallel research
    - When research is complete
    
    Args:
        state: Current supervisor state with messages and research progress
        
    Returns:
        Command to proceed to supervisor_tools node with updated state
    """

    supervisor_messages = state.get("supervisor_messages", []) # check existing messages shared with supervisor

    lead_researcher_system_prompt = lead_researcher_prompt.format(
        date=date_today_str(),
        max_concurrent_research_units=max_concurrent_researchers,
        max_researcher_iterations=max_research_iterations
    )

    messages = [SystemMessage(content=lead_researcher_system_prompt)] + supervisor_messages

    response = await supervisor_model.ainvoke(messages)

    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

async def supervisor_tools(state: SupervisorState) -> Command[Literal["supervisor", "__end__"]]:
    """
    Execute supervisor decisions - either conduct research or end the process.
    
    Handles:
    - Executing think_tool calls for strategic reflection
    - Launching parallel research agents for different topics
    - Aggregating research results
    - Determining when research is complete
    
    Args:
        state: Current supervisor state with messages and iteration count
        
    Returns:
        Command to continue supervision, end process, or handle errors
    """

    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]

    # Initialize variables for single return pattern
    tool_messages = []
    raw_notes_batch = []
    next_step = "supervisor"  # Default next step
    should_end = False

    ##### EXIT CRITERIA CHECKS
    exceeded_iterations = research_iterations >= max_research_iterations
    no_tool_calls = not most_recent_message.tool_calls # if no tool calls, treat as exit criteria to prevent infinite loops
    research_complete = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    ) # if ResearchComplete tool was called, treat as exit criteria

    if exceeded_iterations or no_tool_calls or research_complete:
        should_end = True
        next_step = END
    
    else:
        try:
            # Separate each tool calls (since we have both synchronous think_tool calls and asynchronous ConductResearch calls in the same message, we need to handle them separately)
            think_tool_calls = [
                tool_call for tool_call in most_recent_message.tool_calls 
                if tool_call["name"] == "think_tool"
            ]
            
            conduct_research_calls = [
                tool_call for tool_call in most_recent_message.tool_calls 
                if tool_call["name"] == "ConductResearch"
            ]

            # Handle think_tool calls (synchronous)
            for tool_call in think_tool_calls:
                observation = think_tool.invoke(tool_call["args"])
                tool_messages.append(
                    ToolMessage(
                        content=observation,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    )
                )

            # Handle ConductResearch calls (asynchronous)
            if conduct_research_calls:
                # Launch parallel research agents
                researcher_invokations = [
                    researcher_agent.ainvoke({
                        "researcher_messages": [
                            HumanMessage(content=tool_call["args"]["research_topic"])
                        ],
                        "research_topic": tool_call["args"]["research_topic"]
                    }) 
                    for tool_call in conduct_research_calls
                ]

                # Wait for all research to complete
                tool_results = await asyncio.gather(*researcher_invokations)

                # Format research results as tool messages
                # Each sub-agent returns compressed research findings in result["compressed_research"]
                # We write this compressed research as the content of a ToolMessage, which allows
                # the supervisor to later retrieve these findings via get_notes_from_tool_calls()
                research_tool_messages = [
                    ToolMessage(
                        content=result.get("compressed_research", "Error synthesizing research report"),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    ) for result, tool_call in zip(tool_results, conduct_research_calls)
                ]
                
                tool_messages.extend(research_tool_messages)

                # Aggregate raw notes from all research
                raw_notes_batch = [
                    "\n".join(result.get("raw_notes", [])) 
                    for result in tool_results
                ]
        
        except Exception as e:
            print(f"Error in supervisor tools: {e}")
            should_end = True
            next_step = END

    # Single return point with appropriate state updates
    if should_end:
        return Command(
            goto=next_step,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )
    else:
        return Command(
            goto=next_step,
            update={
                "supervisor_messages": tool_messages,
                "raw_notes": raw_notes_batch
            }
        )
    
######### STATE GRAPH DEFINITION ##########
supervisor_graph = StateGraph(SupervisorState)

supervisor_graph.add_node("supervisor", supervisor)
supervisor_graph.add_node("supervisor_tools", supervisor_tools)

supervisor_graph.add_edge(START, "supervisor") # loop back to supervisor for next iteration of coordination

supervisor_agent = supervisor_graph.compile()