import random
import timeit
import string
import copy

def check_history(history):
    labels = ['user', 'system']
    for i,turn in enumerate(history):
        if labels[i%2] != next(iter(turn)):
            return False
    return True

def get_resp(history):
    def f(length=7):
        letters_and_digits = string.ascii_letters + string.digits
        result_str = ''.join(random.choice(letters_and_digits) for _ in range(length))
        return result_str
    
    return f() if check_history(history) else "check history failed"
    
nTurns = 3
task = "baking a cake"
persona = "type1"

personas = {"normal":"You are a helpful AI assistant.",
                "type1":f"Type1 stuff {task}.",
                "type2":f"Type2 stuff {task}."}

frame0 = [{"user":f"You are a helpful AI assistant. Ask questions one at a time. You can ask up to {nTurns} questions in total."}]
frame1 = [{"user":f"{personas[persona]} "}]
nResp = get_resp(frame0)
frame0.append({"system":nResp})
frame1[0]["user"]+=nResp

for _ in range(nTurns):
    nResp = get_resp(frame1)
    frame0.append({"user":nResp})
    frame1.append({"system":nResp})
    nResp = get_resp(frame0)
    frame0.append({"system":nResp})
    frame1.append({"user":nResp})

frame0.pop()
frame1.pop()
print(frame0)
print(frame1)