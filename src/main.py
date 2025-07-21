from pydantic import BaseModel

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.agent.agent import agent
from src.agent.agent import AgentState

class QueryRequest(BaseModel):
    query: str

app: FastAPI = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state: AgentState = {
    "user_message": "",
    "chat_history": [],
    "context": [],
    "answer": ""
}

@app.get('/')
def root_message():
    return {"message": "Sakinah Backend System running"}

@app.post('/query')
async def user_query(query: QueryRequest):
    print(f"Received query: {query.query}")
    state["user_message"] = query.query
    response = agent(state)
    return {"text": response}