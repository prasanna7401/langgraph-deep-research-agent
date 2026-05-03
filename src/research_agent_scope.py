
"""User Clarification, Research Brief Generation, and Human Review.

This module implements the scoping phase of the research workflow, where we:
1. Assess if the user's request needs clarification
2. Generate a detailed research brief from the conversation
3. Pause for a human reviewer to either approve the brief or request refinements

The workflow uses structured output to make deterministic decisions about
whether sufficient context exists to proceed with research, and a dynamic
``interrupt()`` to gate the brief on human approval before downstream agents
consume it.
"""

from typing_extensions import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

from src.agent_prompts import (
    clarify_with_user_instructions,
    transform_messages_into_research_topic_prompt,
    revise_research_brief_prompt,
)
from src.state_scope import AgentInputState, AgentState, ClarifyUserRequest, ResearchBrief
from src.utils import date_today_str

model = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

############### NODE FUNCTIONS ###############

def clarify_user_request(state: AgentState) -> Command[Literal["write_research_brief", "__end__"]]:
    """
    Determine if the user's request contains sufficient information to proceed with research.

    Uses structured output to make deterministic decisions and avoid hallucination.
    Routes to either research brief generation or ends with a clarification question.
    """
    clarify_model = model.with_structured_output(ClarifyUserRequest)

    clarification_prompt = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages=state["messages"]),
        date=date_today_str()
    )

    response = clarify_model.invoke([HumanMessage(content=clarification_prompt)])


    # Based on the structured output, decide whether to proceed to research brief generation or ask for clarification
    if response.need_clarification:
        return Command(
            goto=END,
            update={
                "messages": [AIMessage(content=response.question_to_user)]
            }
        )
    else:
        return Command(
            goto="write_research_brief",
            update={
                "messages": [AIMessage(content=response.verification)],
            }
        )


def write_research_brief(state: AgentState):
    """
    Generate or revise the research brief.

    Dual-mode: if ``reviewer_feedback`` is present in state alongside an existing
    ``research_brief``, tweak the prior brief using the revise prompt instead of
    regenerating from the full conversation. Otherwise, run the original
    transform-from-messages flow. Feedback is consumed and cleared in the same
    return so the loop resets cleanly on the next human review.
    """
    research_brief_model = model.with_structured_output(ResearchBrief)

    feedback = state.get("reviewer_feedback")
    previous_brief = state.get("research_brief")

    if feedback and previous_brief:
        prompt = revise_research_brief_prompt.format(
            previous_brief=previous_brief,
            reviewer_feedback=feedback,
            date=date_today_str(),
        )
    else:
        prompt = transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=date_today_str(),
        )

    response = research_brief_model.invoke([HumanMessage(content=prompt)])

    return {
        "research_brief": response.research_brief,
        "supervisor_messages": [HumanMessage(content=f"{response.research_brief}")],
        "reviewer_feedback": None,  # consume and clear so the loop resets
    }


def human_review_brief(state: AgentState) -> Command[Literal["write_research_brief", "__end__"]]:
    """
    Pause for a human reviewer. Approve to proceed; reject with feedback to loop
    back to ``write_research_brief`` for a targeted revision.

    Resume payload contract:
        ``Command(resume={"approved": True})`` -> graph proceeds to END.
        ``Command(resume={"approved": False, "feedback": "<text>"})`` -> stores
        feedback in state and re-enters ``write_research_brief`` in revise mode.
    """
    review = interrupt({
        "type": "brief_review",
        "research_brief": state["research_brief"],
        "instructions": "Approve to proceed, or provide recommendations to refine the brief.",
    })

    if review.get("approved"):
        return Command(goto=END)

    feedback = review.get("feedback") or "Please refine the brief."
    return Command(
        goto="write_research_brief",
        update={"reviewer_feedback": feedback},
    )

############### GRAPH CONSTRUCTION ###############

deep_research_builder = StateGraph(AgentState, input_schema=AgentInputState)

deep_research_builder.add_node("clarify_user_request", clarify_user_request)
deep_research_builder.add_node("write_research_brief", write_research_brief)
deep_research_builder.add_node("human_review_brief", human_review_brief)

deep_research_builder.add_edge(START, "clarify_user_request")
deep_research_builder.add_edge("write_research_brief", "human_review_brief")

scope_research_agent = deep_research_builder.compile()
