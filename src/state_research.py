
"""
State Definitions and Pydantic Schemas for Research Agent

This module defines the state objects and structured schemas used for
the research agent workflow, including researcher state management and output schemas.
"""

import operator
from typing import TypedDict, Annotated, List, Sequence
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

########## STATE DEFINITIONS ##########

class ResearcherState(TypedDict):
    """
    State for the research agent containing message history and research metadata.

    This state tracks the researcher's conversation, iteration count for limiting
    tool calls, the research topic being investigated, compressed findings,
    and raw research notes for detailed analysis.
    """
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages] # A sequence of messages representing the conversation history of the researcher.
    tool_call_iterations: int
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]

class ResearcherOutputState(TypedDict):
    """
    Output state for the research agent containing final research results.

    This represents the final output of the research process with compressed
    research findings and all raw notes from the research process.
    """
    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]

########## OUTPUT SCHEMAS ##########

class ClarifyUserRequest(BaseModel): # From Scope Agent
    """Structured output for requesting clarification from the user."""
    need_clarification: Annotated[bool, Field(..., description="Indicates whether clarification is needed from the user.")] 
    question_to_user: Annotated[str, Field(default="", description="The question to ask the user for clarification. Required only when need_clarification=True; otherwise empty string.")]
    verification: Annotated[str, Field(default="", description="Acknowledgment message confirming the user's request is sufficient to proceed with research. Required only when need_clarification=False; otherwise empty string.")]

class ResearchBrief(BaseModel): # From Scope Agent
    """Structured output for the research brief generated from user input."""
    research_brief: Annotated[str, Field(..., description="A concise research brief summarizing the user's request and the research objectives to guide the research agents.")]

class WebSearchSummary(BaseModel):
    """Schema for webpage content summarization."""
    summary: str = Field(description="Concise summary of the webpage content")
    key_points: str = Field(description="Important quotes and excerpts from the content")
