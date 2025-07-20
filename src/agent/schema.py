from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    user_message: str  #the message entered by the user
    context: str # context the is  built using rag
    requires_revision: bool # check whether the retrieved information is relevant to user message. if not retry it
    chat_history: list[str] # history of chat between the agent and the user
    answer: str # generated the answer by the LLM