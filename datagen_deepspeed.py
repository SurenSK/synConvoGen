import mii

pipe = mii.pipeline("cognitivecomputations/dolphin-2.9-llama3-8b")
response = pipe(["DeepSpeed is", "Seattle is"], max_new_tokens=128)
print(response)