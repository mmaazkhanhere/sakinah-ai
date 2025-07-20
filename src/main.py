from fastapi import FastAPI
from pydantic import BaseModel

class UserQuery(BaseModel):
    query: str

app: FastAPI = FastAPI()


@app.get('/')
def root_message():
    return {"message": "Sakinah Backend System running"}

@app.post('/query')
async def user_query(query: UserQuery):
    # pass the user query to langgraph agent
    return {"response": "This is a placeholder response for the user query: " + query.query}