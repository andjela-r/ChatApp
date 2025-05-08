from fastapi import FastAPI
import httpx
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js
        "http://localhost:8001"
        ],
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
        reply = "That's what she said. Just kidding."
        f"You said: {request.message}"
    else:
        reply = f"Gasp. You said: {request.message}"
    return ChatResponse(response=reply)


@app.post("/inference", response_model=ChatResponse)
async def inference(request: ChatRequest):
    print("Sent message: ", request.message)
    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(
                "http://smollm2_model:8001/predict",  # model FastAPI endpoint
                json={
                    "message": request.message,
                    "personality": request.personality
                }
            )
            res.raise_for_status()
            result = res.json()
            print("Response: ", result)
            return ChatResponse(response=result["response"])
        except Exception as e:
            return ChatResponse(response=f"Error: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Hello World"}
