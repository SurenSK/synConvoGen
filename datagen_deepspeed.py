import mii
pipe = mii.pipeline("mistralai/Mistral-7B-v0.1", token="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")
response = pipe(["DeepSpeed is", "Seattle is"], max_new_tokens=128)
print(response)