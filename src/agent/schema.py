from typing import Annotated, Dict
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    user_message: str  #the message entered by the user
    context: list[str] # context the is  built using rag
    chat_history: list[Dict[str, str]] # history of chat between the agent and the user
    answer: str # generated the answer by the LLM