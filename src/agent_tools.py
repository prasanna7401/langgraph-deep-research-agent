from concurrent.futures import ThreadPoolExecutor
from typing import List, Literal, Annotated

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, InjectedToolArg
from tavily import TavilyClient

# custom imports
from src.agent_prompts import summarize_webpage_prompt
from src.state_research import WebSearchSummary
from src.utils import date_today_str

########## INITIALIZATION ##########

model = ChatAnthropic(model="claude-haiku-4-5", temperature=0)
tavily_client = TavilyClient()

########### WEB SEARCH SETUP (Helper Functions for Tavily Search) ##########

def tavily_search_multiple(
    search_queries: List[str], 
    max_results: int = 3, 
    topic: Literal["general", "news", "finance"] = "general", 
    include_raw_content: bool = True, 
) -> List[dict]:
    """
    Perform search using Tavily API for multiple queries.

    Args:
        search_queries: List of search queries to execute
        max_results: Maximum number of results per query
        topic: Topic filter for search results
        include_raw_content: Whether to include raw webpage content

    Returns:
        List of search result dictionaries
    """

    search_docs = []
    for query in search_queries:
        result = tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic
        )
        search_docs.append(result)
    return search_docs


def summarize_webpage_content(webpage_content: str) -> str:
    """
    Summarize webpage content using the configured summarization model.

    Args:
        webpage_content: Raw webpage content to summarize

    Returns:
        Formatted summary with key excerpts
    """
    try:
        summarization_model = model.with_structured_output(WebSearchSummary)
        webpage_summarization_prompt = summarize_webpage_prompt.format(
            webpage_content=webpage_content,
            date=date_today_str()
        )
        summary = summarization_model.invoke([
            HumanMessage(content=webpage_summarization_prompt)
        ])

        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_points>\n{summary.key_points}\n</key_points>"
        )

        return formatted_summary

    except Exception as e:
        print(f"Failed to summarize webpage: {str(e)}")
        return webpage_content[:500] + "..." if len(webpage_content) > 500 else webpage_content

def deduplicate_search_results(search_results: List[dict]) -> dict:
    """
    Deduplicate search results by URL to avoid processing duplicate content.

    Args:
        search_results: List of search result dictionaries

    Returns:
        Dictionary mapping URLs to unique results
    """
    unique_results = {}

    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = result
    return unique_results

def process_search_results(unique_results: dict) -> dict:
    """
    Process search results by summarizing content where available.

    Summarizations run in parallel via ThreadPoolExecutor: each raw_content
    page triggers one Anthropic LLM call, and waiting for them sequentially
    used to dominate every tavily_search invocation.

    Args:
        unique_results: Dictionary of unique search results

    Returns:
        Dictionary of processed results with summaries
    """
    if not unique_results:
        return {}

    items = list(unique_results.items())

    def process_one(item: tuple[str, dict]) -> tuple[str, dict]:
        url, result = item
        if not result.get("raw_content"):
            content = result['content']
        else:
            content = summarize_webpage_content(result['raw_content'])
        return url, {'title': result['title'], 'content': content}

    with ThreadPoolExecutor(max_workers=min(8, len(items))) as executor:
        processed = list(executor.map(process_one, items))

    return dict(processed)

def format_search_output(summarized_results: dict) -> str:
    """
    Format search results into a well-structured string output.

    Args:
        summarized_results: Dictionary of processed search results

    Returns:
        Formatted string of search results with clear source separation
    """
    if not summarized_results:
        return "No valid search results found. Please try different search queries or use a different search API."

    formatted_output = "Search results: \n\n"

    for i, (url, result) in enumerate(summarized_results.items(), 1):
        formatted_output += f"\n\n--- SOURCE {i}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "-" * 80 + "\n"

    return formatted_output

############### RESEARCH AGENT TOOLS ###############

# NOTE: `max_results` and `topic` are wrapped in `Annotated[..., InjectedToolArg]`.
# `InjectedToolArg` hides the argument from the tool schema sent to the LLM, so the
# model only sees `query` and cannot override our defaults at runtime. The values
# are still injected by LangGraph at execution time. Use this whenever a tool
# parameter is operator-controlled (budgets, runtime config, secrets) rather than
# something the LLM should choose.
@tool(parse_docstring=True)
def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 3,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> str:
    """
    Fetch results from Tavily search API with content summarization.

    Args:
        query: A single search query to execute
        max_results: Maximum number of results to return
        topic: Topic to filter results by ('general', 'news', 'finance')

    Returns:
        Formatted string of search results with summaries
    """
    search_results = tavily_search_multiple(
        [query],  # Convert single query to list for the internal function
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
    )

    # Deduplicate results by URL to avoid processing duplicate content
    unique_results = deduplicate_search_results(search_results)

    # Process results with summarization
    summarized_results = process_search_results(unique_results)

    # Format output for consumption
    return format_search_output(summarized_results)

@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"
