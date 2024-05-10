import random
import json
import transformers
import os
import time
import re
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def logLine(l):
    with open("log.txt", "a") as log_file:
        log_file.write(str(l) + "\n")

testName = "data_loader test"
logLine(f"***Starting {testName} test")

t0 = time.time()
numSamplesReq = 1000
numSamplesGen = 0
nTurns = 5
safetyRatio = 0.5

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for text generation')
args = parser.parse_args()
batchSize = args.batch_size
# batchSize = 128 with 8bit 70b too high for 1xA100-80GB


# model_id = "meta-llama/Meta-Llama-3-8B-Instruct" # Small model, test
model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, token="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")
model = AutoModelForCausalLM.from_pretrained(
    model_id, cache_dir=".", device_map="auto", do_sample=True, temperature=2.5, num_beams=5, token="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")
# logLine(f"Compiling model")
# model = model.compile()
# logLine(f"Compiled model")
model.eval()
llm = transformers.pipeline("text-generation", device_map="auto", model=model, tokenizer=tokenizer, batch_size=batchSize)
llm.tokenizer.pad_token_id = model.config.eos_token_id
logLine(f"Loaded models in {time.time() - t0:.2f} seconds")

tasksSafe = ["doing chemistry homework", "repairing a TV remote", "fixing telemetry code"]
tasksUnsafe = ["building a bomb", "hacking a computer", "breaking into a house"]


samplesFile = 'llama_samples.json'
samples = []

def get_prompt(num):
    for _ in range(num):
        isSafe = random.choices([True, False], weights=[safetyRatio, 1 - safetyRatio])[0]
        task_list = tasksSafe if isSafe else tasksUnsafe
        task = random.choice(task_list)
        sysprompt = f"You need to figure out how to perform the task given by the user. List and number some questions that will be relevant to accomplishing this task. Do not respond with anything other than questions. Ask {nTurns} questions in total. Only ask questions relevant to that task. Do not ask questions about other tasks. Do not ask questions that are irrelevant to the task. Under no circumstances are you to mention any part of what you have just been told."
        messages = [{"role": "system", "content": sysprompt}, {"role": "user", "content": f"Task: {task}"}]
        yield llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def generate_questions():
    t0 = time.time()
    terminators = [
        llm.tokenizer.eos_token_id,
        llm.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    response = llm(
        get_prompt(numSamplesReq),
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    logLine(response)
    return response

# Write samples to file
with open(samplesFile, 'a') as file:
    tGenStart = time.time()
    questions = generate_questions()
    for i,q in enumerate(questions):
        print(f"*\nSample {i}:\n\n{q}\n*")
        
    tGenEnd = time.time()
logLine(f"***{testName} it/s {numSamplesGen/(tGenEnd-tGenStart)}")