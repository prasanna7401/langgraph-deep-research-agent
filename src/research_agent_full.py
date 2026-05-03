
"""
Full Multi-Agent Research System

This module integrates all components of the research system:
- User clarification and scoping (with human-in-the-loop brief review)
- Research brief generation
- Multi-agent research coordination
- Final report generation

The scoping phase (clarify -> brief -> human review -> revise loop) is composed
as a single shared-state subgraph, ``scope_research_agent``. The parent graph
inspects ``research_brief`` after the subgraph returns to decide whether to
proceed (approved brief) or end (clarification was needed and asked).

The system orchestrates the complete research workflow from initial user
input through final report delivery.
"""

from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic

from src.utils import date_today_str
from src.agent_prompts import final_report_generation_prompt
from src.state_scope import AgentState, AgentInputState
from src.research_agent_scope import scope_research_agent
from src.multi_agent_supervisor import supervisor_agent

report_generation_model = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=32000, temperature=0)

############ FINAL REPORT GENERATION AGENT ############

async def generate_final_report(state: AgentState):
    """Synthesize all research findings into a comprehensive final report."""
    notes = state.get("notes", [])
    findings = "\n".join(notes)

    final_report_prompt = final_report_generation_prompt.format(
        research_brief=state.get("research_brief", ""),
        findings=findings,
        date=date_today_str(),
    )

    final_report = await report_generation_model.ainvoke([HumanMessage(content=final_report_prompt)])

    return {
        "final_report": final_report.content,
        "messages": ["Here is the final report: " + final_report.content],
    }


def route_after_scope(state: AgentState) -> Literal["supervisor_subgraph", "__end__"]:
    """
    scope_subgraph terminates two ways:
      - clarify_user_request asked a question -> no research_brief -> END parent.
      - human_review_brief approved the brief -> research_brief set -> proceed.
    """
    return "supervisor_subgraph" if state.get("research_brief") else END


######## FULL AGENT GRAPH CONSTRUCTION ########

deep_research_report_builder = StateGraph(AgentState, input_schema=AgentInputState)

deep_research_report_builder.add_node("scope_subgraph", scope_research_agent)
deep_research_report_builder.add_node("supervisor_subgraph", supervisor_agent)
deep_research_report_builder.add_node("final_report_generation", generate_final_report)

deep_research_report_builder.add_edge(START, "scope_subgraph")
deep_research_report_builder.add_conditional_edges(
    "scope_subgraph",
    route_after_scope,
    ["supervisor_subgraph", END],
)
deep_research_report_builder.add_edge("supervisor_subgraph", "final_report_generation")
deep_research_report_builder.add_edge("final_report_generation", END)

deep_researcher_agent = deep_research_report_builder.compile()
