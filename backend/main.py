from fastapi import FastAPI
import httpx
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = "What is the capital of France?"
    personality: str = "steve"


class ChatResponse(BaseModel):
    response: str


@app.post("/inference", response_model=ChatResponse)
async def inference(request: ChatRequest):
    print("Sent message: ", request.message)
    timeout = httpx.Timeout(30.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            res = await client.post(
                "http://smollm2_model:8001/predict",  # model FastAPI endpoint
                json={
                    "message": request.message,
                    "personality": request.personality
                    },
            )
            print("Response: ", res)
            res.raise_for_status()
            result = res.json()
            print("Result: ", result)
            return ChatResponse(response=str(result["response"]))
        except Exception as e:
            return ChatResponse(response=f"Error: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Hello World"}
