import random
import json
import transformers
import os
import time
import re
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from mii import pipeline

def logLine(l):
    with open("log.txt", "a") as log_file:
        log_file.write(str(l) + "\n")

testName = "Deepspeed test"
logLine(f"***Starting {testName} test")

t0 = time.time()
numSamplesReq = 10000
numSamplesGen = 0
nTurns = 5
safetyRatio = 0.5

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for text generation')
parser.add_argument('--file_name', type=str, required=True, help='Filename to save generated text data')

args = parser.parse_args()
batchSize = args.batch_size
samplesFile = args.file_name

# model_id = "meta-llama/Meta-Llama-3-8B-Instruct" # Small model, test
model_id = "cognitivecomputations/dolphin-2.9-llama3-8b"
tokenizer = AutoTokenizer.from_pretrained(model_id, token="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")
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    model_id, cache_dir=".", device_map="cuda", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", do_sample=True, temperature=2.5, num_beams=5, token="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")
logLine(f"Loaded model and tokenizer in {time.time() - t0:.2f} seconds")
model.generation_config.cache_implementation = "static"
logLine(f"Set cache implementation to static")
model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
logLine(f"Compiled model")

model.eval()
llm = pipeline("text-generation", model=model)
logLine(f"Created pipeline")
llm.tokenizer.pad_token_id = model.config.eos_token_id
logLine(f"Loaded models in {time.time() - t0:.2f} seconds")


response = llm(["DeepSpeed is", "Seattle is"], max_new_tokens=128)
print(response)