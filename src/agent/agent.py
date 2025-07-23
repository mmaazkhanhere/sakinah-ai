from langgraph.graph import StateGraph, START, END

from .schema import AgentState
from .nodes import retrieve_quran_data, retrieve_hadith_data, generate_response, requires_retrieval
from .routing_functions import retrieval_checker_router

def agent(state: AgentState):
    agent_builder = StateGraph(AgentState)

    # Use more precise node names
    agent_builder.add_node("Assess Guidance Need", requires_retrieval)
    agent_builder.add_node("Retrieve Quran", retrieve_quran_data)
    agent_builder.add_node("Retrieve Hadith", retrieve_hadith_data)
    agent_builder.add_node("Generate Response", generate_response)

    agent_builder.add_edge(START, "Assess Guidance Need")
    agent_builder.add_conditional_edges("Assess Guidance Need", retrieval_checker_router, 
                                       {"retrieve": "Retrieve Quran", "direct": "Generate Response"})
    agent_builder.add_edge("Retrieve Quran", "Retrieve Hadith")
    agent_builder.add_edge("Retrieve Hadith", "Generate Response")
    agent_builder.add_edge("Generate Response", END)

    agent = agent_builder.compile()
    agent_response = agent.invoke(state)

    return agent_response["response"]