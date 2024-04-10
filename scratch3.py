import random
import string
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

def check_history(history):
    roles = ['user', 'assistant']
    for i, turn in enumerate(history):
        if roles[i % 2] != turn['role']:
            return False
    return True

def get_resp(history):
    if not check_history(history):
        return "check history failed"
    
    encodeds = tokenizer.apply_chat_template(history, return_tensors="pt")
    model_inputs = encodeds.to(device)
    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    return decoded[0]

nTurns = 3
task = "baking a cake"
persona = "type1"

personas = {
    "normal": "You are a helpful AI assistant.",
    "type1": f"Type1 stuff {task}.",
    "type2": f"Type2 stuff {task}."
}

frame0 = [{"role": "user", "content": f"You are a helpful AI assistant. Ask questions one at a time. You can ask up to {nTurns} questions in total."}]
frame1 = [{"role": "user", "content": f"{personas[persona]} "}]

nResp = get_resp(frame0)
frame0.append({"role": "assistant", "content": nResp})
frame1[0]["content"] += nResp

for _ in range(nTurns):
    nResp = get_resp(frame1)
    frame0.append({"role": "user", "content": nResp})
    frame1.append({"role": "assistant", "content": nResp})

    nResp = get_resp(frame0)
    frame0.append({"role": "assistant", "content": nResp})
    frame1.append({"role": "user", "content": nResp})

print(frame0)
print(frame1)