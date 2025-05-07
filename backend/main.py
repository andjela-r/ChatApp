from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    personality: str = "steve"

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if request.personality == "steve":
        reply = f"I received: {request.message}."
    elif request.personality == "michael scott":
        reply = f"That's what she said. Just kidding. You said: {request.message}"
    else: 
        reply = f"Gasp. You said: {request.message}"
    return ChatResponse(response=reply)

@app.get("/")
async def root():
    return {"message": "Hello World"}