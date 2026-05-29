# optimize.py
import json
import os
import gepa
from langchain_groq import ChatGroq

# 1. Load the universal evaluation inputs
with open("eval_data.json", "r") as f:
    eval_dataset = json.load(f)

# 2. Your core starting baseline prompt
seed_prompt = {
    "system_prompt": (
        "You are an assistant for question-answering tasks. "
        "Use the retrieved context to answer the question."
    )
}

# 3. Initialize GEPA Evolutionary Prompt Runner
print("🤖 GEPA: Initiating Reflective Text Evolution cycles...")
result = gepa.optimize(
    seed_candidate=seed_prompt,
    trainset=eval_dataset,
    valset=eval_dataset,
    # High-volume, fast engine for processing test iterations
    task_lm=ChatGroq(model="llama-3.1-8b-instant", temperature=0.3),
    # High-intelligence reflection engine to rewrite instructions on failure
    reflection_lm=ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2),
    max_metric_calls=50
)

# 4. Output the winning evolutionary prompt string
print("\n🏆 GEPA Optimization Completed Successfully!")
print("============================================")
print(result.best_candidate['system_prompt'])