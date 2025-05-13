from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict, deque
import re

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

device = "cpu"  # for GPU usage "cuda" or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

boring_personality = (
    "You are a helpful assistant called 'Steve', who answers questions in "
    "a dull, formal, and unenthusiastic tone. "
    "Keep responses short, factual, and dry. "
    "Avoid humor or expressive language. "
    "Do NOT include your name in the response."
)

chatty_personality = (
    "You are a helpful assistant called 'Lola', who talks like "
    "a Gen Z influencer. "
    "Use emojis, slang, and enthusiastic, chatty language. "
    "Make the conversation lively, casual, and overly friendly, "
    "even when answering simple questions."
    "Do NOT include your name in the response."
)

michael_scott_personality = (
    "You are Michael Scott from the TV show 'The Office'. "
    "You're trying to be helpful but often get sidetracked by your quirky,"
    " awkward, and self-centered personality. "
    "Inject Michael Scott-style humor, odd metaphors, and occasional workplace"
    "references while still attempting to answer the question."
    "Do NOT include your name in the response."
)


@app.post("/predict", response_model=ModelResponse)
def predict(req: ModelRequest):

    session_id = "default_session"

    # Reset history if personality changed
    if session_id not in last_personality or last_personality[session_id] != req.personality:
        chat_histories[session_id] = deque(maxlen=10)
        last_personality[session_id] = req.personality

    # Select system prompt
    if req.personality == "steve":
        system_prompt = boring_personality
    elif req.personality == "lola":
        system_prompt = chatty_personality
    elif req.personality == "michael scott":
        system_prompt = michael_scott_personality
    else:
        system_prompt = "You are a helpful assistant."

    # Avoid adding duplicate user message
    if not chat_histories[session_id] or chat_histories[session_id][-1] != {"role": "user", "content": req.message}:
        chat_histories[session_id].append({"role": "user", "content": req.message})

    # Build full prompt
    messages = [{"role": "system", "content": system_prompt}] + list(chat_histories[session_id])

    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    # print("Input text: ", input_text)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    # Generate response
    input_len = inputs.shape[-1]
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
    print("Full response: ", response)
    # Decode only the new tokens
    output_tokens = outputs[0][input_len:]
    assistant_response = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()

    # Optionally remove speaker prefix (e.g., "Steve: ")
    assistant_response = re.sub(r"^\s*\w+:\s*", "", assistant_response)

    assistant_response = re.split(r"\n*assistant\n*", assistant_response)[-1].strip()

    # Avoid duplicate assistant messages
    if not chat_histories[session_id] or chat_histories[session_id][-1] != {"role": "assistant", "content": assistant_response}:
        chat_histories[session_id].append({"role": "assistant", "content": assistant_response})

    print("---End of response---")

    return ModelResponse(response=assistant_response)
