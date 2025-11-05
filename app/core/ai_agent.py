from typing import Dict, List

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import inspect


ROLE_TO_MSG = {
    "user": HumanMessage,
    "assistant": AIMessage,
    "system": SystemMessage,
}


def _create_react_agent_with_prompt(model, tools, system_prompt: str):
    """
    """
    params = inspect.signature(create_react_agent).parameters
    if "state_modifier" in params:
        return create_react_agent(model=model, tools=tools, state_modifier=system_prompt)
    elif "messages_modifier" in params:
        return create_react_agent(model=model, tools=tools, messages_modifier=system_prompt)
    else:
        # Fallback: no modifier supported in this version
        return create_react_agent(model=model, tools=tools)


def get_response_from_ai_agents(
    llm_id: str,
    messages: List[Dict[str, str]],      # [{"role":"user|assistant|system","content":"..."}]
    allow_search: bool,
    system_prompt: str,
) -> str:
    llm = ChatGroq(model=llm_id)  # GROQ_API_KEY must be in env
    tools = [TavilySearch(max_results=2)] if allow_search else []

    agent = _create_react_agent_with_prompt(llm, tools, system_prompt)

    # Normalize incoming messages -> LangChain BaseMessage list
    lc_msgs = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        cls = ROLE_TO_MSG.get(role, HumanMessage)
        lc_msgs.append(cls(content=content))

    # Run the agent
    result = agent.invoke({"messages": lc_msgs})

    # Extract the last assistant reply
    out = ""
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage):
            out = msg.content
            break
        if isinstance(msg, dict) and msg.get("role") in ("assistant", "ai"):
            out = msg.get("content", "")
            break
    return out
