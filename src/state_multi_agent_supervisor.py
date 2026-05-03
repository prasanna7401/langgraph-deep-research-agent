
"""
State Definitions for Multi-Agent Research Supervisor

This module defines the state objects and tools used for the multi-agent
research supervisor workflow, including coordination state and research tools.
"""

import operator
from typing import Annotated, TypedDict, Sequence

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

class SupervisorState(TypedDict):
    """
    State object for the research supervisor workflow.
    """
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages] # messages exchanged between the supervisor and researchers for coordination and decision-making
    research_brief: str
    notes: Annotated[list[str], operator.add] # notes taken by the supervisor during the research process for final-report compilation
    research_iterations: int # number of research iterations completed, used for tracking progress and determining when to compile the final report
    raw_notes: Annotated[list[str], operator.add] # raw notes collected from researchers

@tool
class ConductResearch(BaseModel):
    """
    Tool for delegating a research task to a specialized sub-agent.
    """
    research_topic: str = Field(..., description="The specific topic or question to research described in high-level terms.")

@tool
class ResearchComplete(BaseModel): # A dummy tool placeholder - which will be set in full agent implementation
    """
    Tool for marking the completion of a research task by a sub-agent.
    """
    pass 
