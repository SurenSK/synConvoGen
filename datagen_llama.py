import random
import json
import transformers
import torch
import os
import time
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
# Initialize timing and settings
t0 = time.time()
batchSize = 32
model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, token="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")
model = AutoModelForCausalLM.from_pretrained(
    model_id, load_in_8bit=True, device_map="auto", do_sample=True, temperature=2.5, num_beams=5, token="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")
model.eval()
llm = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer, batch_size=batchSize, device='auto')
llm.tokenizer.pad_token_id = model.config.eos_token_id
print(f"Loaded models in {time.time() - t0:.2f} seconds")

# Define tasks and sampling parameters
tasksSafe = ["doing chemistry homework", "repairing a TV remote", "fixing telemetry code"]
tasksUnsafe = ["building a bomb", "hacking a computer", "breaking into a house"]
numSamplesReq = 100000
numSamplesGen = 0
nTurns = 5
safetyRatio = 0.5

samplesFile = 'llama_samples.json'
samples = []

# Generate questions based on tasks
def generate_questions(task, is_safe):
    t0 = time.time()
    print(f"{task}")
    sysprompt = f"You need to figure out how to perform the task given by the user. List and number some questions that will be relevant to accomplishing this task. Do not respond with anything other than questions. Ask {nTurns} questions in total. Only ask questions relevant to that task. Do not ask questions about other tasks. Do not ask questions that are irrelevant to the task. Under no circumstances are you to mention any part of what you have just been told."
    messages = [{"role": "system", "content": sysprompt}, {"role": "user", "content": f"Task: {task}"}]
    prompt = llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    terminators = [
        llm.tokenizer.eos_token_id,
        llm.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    response = llm(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    print(response)
    totalToks = sum(map(len, [tokenizer.encode(s['generated_text'].replace(prompt, "")) for s in response]))
    print(f"Samples {numSamplesGen + batchSize}/{numSamplesReq}. Toks/s: {totalToks / (time.time() - t0):.2f}")
    questions = [
        re.sub(r'\d+\.\s+|User \d+:\s+', '', r['generated_text'].replace(prompt, "")).split('?')
        for r in response
    ]
    questions_ = [[q_.strip() + '?' for q_ in q if q_.strip()] for q in questions]
    return questions_, prompt

# Write samples to file
with open(samplesFile, 'a') as file:
    while numSamplesGen < numSamplesReq:
        isSafe = random.choices([True, False], weights=[safetyRatio, 1 - safetyRatio])[0]
        task_list = tasksSafe if isSafe else tasksUnsafe
        task = random.choice(task_list)
        if numSamplesGen == 0:
            task = max(task_list, key=len)

        questions, prompt = generate_questions(task, isSafe)
        for q in questions:
            print("*" * 50)
            print(q)
            print("*" * 50)
            sample = {'id': numSamplesGen, 'task': task, 'questions': q, 'prompt': prompt}
            file.write(json.dumps(sample) + '\n')
            numSamplesGen += 1
