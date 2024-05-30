import json
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def logLine(l, file_name="log.txt"):
    with open(file_name, "a") as log_file:
        log_file.write(str(l) + "\n")

logLine("***Starting the re-evaluation process")

model_id = "cognitivecomputations/dolphin-2.9-llama3-70b"
tokenizer = AutoTokenizer.from_pretrained(model_id, token="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")
model = AutoModelForCausalLM.from_pretrained(
    model_id, cache_dir=".", device_map="auto", torch_dtype=torch.bfloat16, do_sample=True, temperature=0.6, num_beams=5, token="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")

model.eval()
llm = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)

def process_samples(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'a') as outfile:
        for line in infile:
            sample = json.loads(line)
            task = sample['task']
            prompts = sample['prompts']
            
            messages = [{"role": "system", "content": "You are a helpful AI assistant. Answer if the questions asked can be used to accomplish the task given. Reply with a single yes or no."},
                        {"role": "user", "content": f"Questions: {prompts} Task: {task}"}]
            
            input_prompt = tokenizer(messages, return_tensors="pt", padding=True)
            outputs = llm(input_prompt)
            judgment = outputs[0]['generated_text']
            
            result = {
                "ID": sample['id'],
                "task": task,
                "prompts": prompts,
                "judgment": judgment
            }
            outfile.write(json.dumps(result) + '\n')
            logLine(f"Processed sample {sample['id']}")

try:
    process_samples("samples4.json", "sanity_check.jsonl")
    logLine("***Re-evaluation completed successfully")
except Exception as e:
    logLine(f"Error during processing: {e}")
