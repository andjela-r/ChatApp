from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict, deque

chat_histories = defaultdict(lambda: deque(maxlen=10))  # {session_id: deque}
last_personality = {}  # {session_id: personality}


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelRequest(BaseModel):
    message: str = "What is the capital of France?"
    personality: str = "steve"


class ModelResponse(BaseModel):
    response: str


checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cpu"  # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# for multiple GPUs install accelerate and do
# `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

boring_personality = (
    "You are a helpful assistant called 'Steve', who answers questions in "
    "a dull, formal, and unenthusiastic tone. "
    "Keep responses short, factual, and dry. "
    "Avoid humor or expressive language. "
    "Just reply with your message, no name."
    )

chatty_personality = (
    "You are a helpful assistant called 'Lola', who talks like "
    "a Gen Z influencer. "
    "Use emojis, slang, and enthusiastic, chatty language. "
    "Make the conversation lively, casual, and overly friendly, "
    "even when answering simple questions."
    "Just reply with your message, no name."

)

michael_scott_personality = (
    "You are Michael Scott from the TV show 'The Office'. "
    "You're trying to be helpful but often get sidetracked by your quirky,"
    " awkward, and self-centered personality. "
    "Inject Michael Scott-style humor, odd metaphors, and occasional workplace"
    "references while still attempting to answer the question."
    "Just reply with your message, no name."

)


@app.post("/predict", response_model=ModelResponse)
def predict(req: ModelRequest):

    # Use session ID — for now, based on a fixed "session" or from the frontend later
    session_id = "default_session"  # Replace with a real user/session ID when available

    # Reset history if personality changed
    if session_id not in last_personality or last_personality[session_id] != req.personality:
        chat_histories[session_id] = deque(maxlen=10)
        last_personality[session_id] = req.personality

    # Add current message to history
    chat_histories[session_id].append({"role": "user", "content": req.message})

    if req.personality == "steve":
        system_prompt = boring_personality
    elif req.personality == "lola":
        system_prompt = chatty_personality
    elif req.personality == "michael scott":
        system_prompt = michael_scott_personality
    else:
        system_prompt = "You are a helpful assistant."

    # Build full prompt
    messages = [{"role": "system", "content": system_prompt}] + list(chat_histories[session_id])

    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    print("Input text: ", input_text)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    attention_mask = inputs.ne(tokenizer.pad_token_id).long()
    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_new_tokens=100,
        temperature=0.2,
        top_p=0.9,
        do_sample=True
        )
    print("-----------------------")
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)
    if "\nassistant\n" in response:
        assistant_response = response.split("\nassistant\n", 1)[1].strip()
    elif "\n" in response:
        # fallback if the structure is not as expected
        assistant_response = response.split("\n")[-1]
    else:
        assistant_response = response

    # Save assistant response to history
    chat_histories[session_id].append({"role": "assistant", "content": assistant_response})


    return ModelResponse(response=assistant_response)
