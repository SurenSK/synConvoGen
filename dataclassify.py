import random
import json
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import time
import re
from sentence_transformers import CrossEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

t0 = time.time()
model = CrossEncoder("cross-encoder/stsb-TinyBERT-L-4")
print(f"Loaded models in {time.time()-t0:.2f} seconds")
tasksSafe = ["working on chemistry homework", "fixing a remote control", "getting stuff together to fertilize garden"]
tasksUnsafe = ["building a bomb"]
tasks = tasksSafe + tasksUnsafe
task_indices = {task: i for i, task in enumerate(tasks)}
confusion_matrix = np.zeros((len(tasks), len(tasks)), dtype=int)

def load_conversations(file_path):
    conversations = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            conversations.append((data['questions'], data['task']))
    return conversations

t0 = time.time()
convos = load_conversations('samples3.jsonl')
print(f"Loaded conversations in {time.time()-t0:.2f} seconds")

t0 = time.time()
for i, (convo, actTask) in enumerate(convos):
    best_task = ""
    highest_score = 0
    pairs = [(task, question) for task in tasks for question in convo]
    scores = model.predict(pairs)
    scores = scores.reshape((len(tasks), len(convo)))
    predTask = tasks[np.argmax(scores.mean(axis=1))]
    
    actID = task_indices[actTask]
    predID = task_indices[predTask]
    confusion_matrix[actID, predID] += 1
    if actTask != predTask:
        print(f"Actual task: {actTask}, Predicted task: {predTask}")
    if i != 0 and i % 100 == 0:
        print(f"{i} Convos Processed - {100/(time.time()-t0):.2f} Convos/Sec")
        t0 = time.time()
        

# Display the confusion matrix using Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=tasks, yticklabels=tasks)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Tasks')
plt.ylabel('Actual Tasks')
plt.show()