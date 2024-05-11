import mii
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", token="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")
pipe = mii.pipeline(model)
response = pipe(["DeepSpeed is", "Seattle is"], max_new_tokens=128)
print(response)