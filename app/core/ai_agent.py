from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.config.settings import settings

ROLE_TO_MSG = {
    "user": HumanMessage,
    "assistant": AIMessage,
    "system": SystemMessage,
}


def get_response_from_ai_agents(llm_id, messages, allow_search, system_prompt):
    llm = ChatGroq(model=llm_id)
    tools = [TavilySearch(max_results=2)] if allow_search else []
    agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)

    lc_msgs = []
    for m in messages:
        # ensure string content
        content = m.content if isinstance(m.content, str) else str(m.content)
        cls = ROLE_TO_MSG.get(m.role, HumanMessage)
        lc_msgs.append(cls(content=content))

    result = agent.invoke({"messages": lc_msgs})

    # extract last assistant reply
    for m in reversed(result.get("messages", [])):
        if isinstance(m, AIMessage):
            return m.content
        if isinstance(m, dict) and m.get("role") in ("assistant","ai"):
            return m.get("content","")
    return ""