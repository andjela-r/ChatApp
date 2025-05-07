from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cpu" # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

boring_personality = "You are a boring helpful assistant."
chatty_personality = "You are a very chatty, Gen Z helpful assistant."
michael_scott_personality = "You are Michael Scott from the TV Series 'The Office' and you are a hhelpful assistant."

def get_response(personality, request):
    if personality == "boring": 
        system_prompt = boring_personality
    elif personality == "chatty":
        system_prompt = chatty_personality
    elif personality == "michael_scott":
        system_prompt = michael_scott_personality
    else:
        system_prompt = "You are a helpful assistant."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request}
        ]

    input_text=tokenizer.apply_chat_template(messages, tokenize=False)
    print(input_text)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
