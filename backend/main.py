from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    personality: str = "default"

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if request.personality == "boring":
        reply = f"I received: {request.message}."
    elif request.personality == "michael_scott":
        reply = f"That's what she said. Just kidding. You said: {request.message}"
    else: 
        reply = f"Gasp. You said: {request.message}"
    return ChatResponse(response=reply)

@app.get("/")
async def root():
    return {"message": "Hello World"}