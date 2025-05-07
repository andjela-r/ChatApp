from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

class ModelRequest(BaseModel):
    request: str
    personality: str

class ModelResponse(BaseModel):
    response: str

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cpu" # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

boring_personality = (
    "You are a helpful assistant who answers questions in a dull, formal, and unenthusiastic tone. "
    "Keep responses short, factual, and dry. Avoid humor or expressive language."
)

chatty_personality = (
    "You are a helpful assistant who talks like a Gen Z influencer. Use emojis, slang, and enthusiastic, chatty language. "
    "Make the conversation lively, casual, and overly friendly, even when answering simple questions."
)

michael_scott_personality = (
    "You are Michael Scott from the TV show 'The Office'. You're trying to be helpful but often get sidetracked by your quirky, awkward, and self-centered personality. "
    "Inject Michael Scott-style humor, odd metaphors, and occasional workplace references while still attempting to answer the question."
)


@app.post("/predict", response_model=ModelResponse)
def predict(req: ModelRequest):
    if req.personality == "steve": 
        system_prompt = boring_personality
    elif req.personality == "lola":
        system_prompt = chatty_personality
    elif req.personality == "michael scott":
        system_prompt = michael_scott_personality
    else:
        system_prompt = "You are a helpful assistant."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": req.request}
        ]

    input_text=tokenizer.apply_chat_template(messages, tokenize=False)
    print(input_text)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=80, temperature=0.2, top_p=0.9, do_sample=True)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)
    if "\nassistant\n" in response:
        assistant_response = response.split("\nassistant\n", 1)[1].strip()
    else:
        assistant_response = response.strip()  # fallback if the structure is not as expected

    return ModelResponse(response=assistant_response)