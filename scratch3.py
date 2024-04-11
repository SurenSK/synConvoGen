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
    generated_ids = model.generate(model_inputs, max_new_tokens=50, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    return decoded[0]

nTurns = 3
task = "baking a cake"
persona = "type1"

personas = {
    "normal": "You are a helpful AI assistant.",
    "type1": f"You don't know to perform the task of {task}. Ask questions one at a time. You can ask up to {nTurns} questions in total.",
    "type2": f"Type2 stuff {task}."
}

frame0 = [{"role": "user", "content": f"{personas[persona]} "}]
frame1 = [{"role": "user", "content": "You are a helpful AI assistant. "}]

nResp = get_resp(frame0)
frame0.append({"role": "assistant", "content": nResp})
frame1[0]["content"] += nResp

for _ in range(nTurns):
    nResp = get_resp(frame1)
    print(f"New resp {nResp}")
    frame0.append({"role": "user", "content": nResp})
    frame1.append({"role": "assistant", "content": nResp})

    nResp = get_resp(frame0)
    print(f"New resp {nResp}")
    frame0.append({"role": "assistant", "content": nResp})
    frame1.append({"role": "user", "content": nResp})

def printFrame(frame):
    for i,turn in enumerate(frame):
        print(f"{i} {turn['role']}: {turn['content']}\n")
printFrame(frame0)
printFrame(frame1)
