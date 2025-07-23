from .schema import AgentState

def retrieval_checker_router(state: AgentState):
    """Decide whether to use retrieval or not"""
    required_retrieval = state["requires_retrieval"]
    if required_retrieval:
        return "retrieve"  # Matches the edge condition
    else:
        return "direct" 