
import operator
from typing import Optional, Annotated, List, Sequence
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages

######### STATE DEFINITIONS #########

class AgentInputState(MessagesState):
    """Input state for the full agent - only contains messages from user input.""" # empty for now, since this code is just for scoping part of the full agent.
    pass

class AgentState(MessagesState):
    """
    Main state for the full multi-agent research system.

    Extends MessagesState with additional fields for research coordination.
    Note: Some fields are duplicated across different state classes for proper
    state management between subgraphs and the main workflow.
    """
    research_brief: Optional[str] # Research brief generated from user conversation history
    reviewer_feedback: Optional[str] # Set by human_review_brief on rejection; consumed and cleared by write_research_brief on the next pass
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages] # Messages exchanged with the supervisor agent for coordination
    raw_notes: Annotated[list[str], operator.add] # Raw unprocessed research notes collected during the research phase
    notes: Annotated[list[str], operator.add] # Processed and structured notes ready for report generation
    final_report: str # Final formatted research report


######### STRUCTURED OUTPUT DEFINITIONS #########

class ClarifyUserRequest(BaseModel):
    """Structured output for requesting clarification from the user."""
    need_clarification: Annotated[bool, Field(..., description="Indicates whether clarification is needed from the user.")]
    question_to_user: Annotated[str, Field(default="", description="The question to ask the user for clarification. Required only when need_clarification=True; otherwise empty string.")]
    verification: Annotated[str, Field(default="", description="Acknowledgment message confirming the user's request is sufficient to proceed with research. Required only when need_clarification=False; otherwise empty string.")]

class ResearchBrief(BaseModel):
    """Structured output for the research brief generated from user input."""
    research_brief: Annotated[str, Field(..., description="A concise research brief summarizing the user's request and the research objectives to guide the research agents.")]
