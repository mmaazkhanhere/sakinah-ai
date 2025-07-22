from langgraph.graph import StateGraph, START, END

from .schema import AgentState
from .nodes import retrieve_quran_data, retrieve_hadith_data, generate_response

def agent(state: AgentState):
    agent_builder = StateGraph(AgentState)

    agent_builder.add_node("Retrieve Quran Data from RAG", retrieve_quran_data)
    agent_builder.add_node("Retrieve Hadith Data from RAG", retrieve_hadith_data)
    agent_builder.add_node("Generate Response", generate_response)

    agent_builder.add_edge(START, "Retrieve Quran Data from RAG")
    # agent_builder.add_edge("Retrieve Data from RAG", "Check Data Relevancy")
    agent_builder.add_edge("Retrieve Quran Data from RAG", "Retrieve Hadith Data from RAG")
    agent_builder.add_edge("Retrieve Hadith Data from RAG", "Generate Response")
    agent_builder.add_edge("Generate Response", END)

    agent = agent_builder.compile()
    agent_response = agent.invoke(state)

    return agent_response["response"]