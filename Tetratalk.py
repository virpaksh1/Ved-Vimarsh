import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
import sys
import os
import random

try:
    import bitsandbytes as bnb
    print("Successfully imported bitsandbytes")
except ImportError:
    print("Error importing bitsandbytes. Attempting to install again...")
    import bitsandbytes as bnb


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Qwen1.5-7B-Chat model - publicly available and efficient to run in Google Colab with T4 GPU
model_name = "Qwen/Qwen1.5-7B-Chat"

print(f"Loading {model_name}...")
start_time = time.time()

# Loading the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Trying to load the model with 4-bit quantization for efficiency
try:
    print("Attempting to load model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance
        device_map="auto",
        trust_remote_code=True,
        quantization_config={"load_in_4bit": True}  # 4-bit quantization for memory efficiency
    )
except Exception as e:
    print(f"4-bit quantization failed with error: {str(e)}")
    print("Falling back to 8-bit quantization...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            load_in_8bit=True  # Try 8-bit quantization instead
        )
    except Exception as e2:
        print(f"8-bit quantization failed with error: {str(e2)}")
        print("Falling back to standard loading (will use more memory)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

load_time = time.time() - start_time
print(f"Model loaded within {load_time:.2f} seconds")

# Define philosophers
philosophers = [
    "Shankaracharya",     # Advaita Vedanta
    "Madhvacharya",       # Dvaita Vedanta
    "Ramanujacharya",     # Vishishtadvaita
    "Charvaka"            # Materialism
]

# Seed with a starting question
conversation = [
    "Shankaracharya: What is ultimately real — the world we perceive, or the eternal Self beyond all names and forms?"
]

def get_next_speaker(current_speaker):
    # Avoid immediate repetition
    remaining = [p for p in philosophers if p != current_speaker]
    return random.choice(remaining)

def generate_response(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt")#.to("cuda")
    output = model.generate(**input_ids, max_new_tokens=150, do_sample=True, temperature=0.8, pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    # Trim the prompt from the output
    response = decoded[len(prompt):].strip()
    # Optional: Clean unwanted repeated speaker tags
    return response.split("\n")[0].strip()

def format_prompt(history, next_speaker):
    prompt = "\n".join(history[-5:]) + f"\n{next_speaker}:"
    return prompt

# Simulate 10 more turns
num_turns = 10
current_speaker = "Shankaracharya"

for _ in range(num_turns):
    next_speaker = get_next_speaker(current_speaker)
    prompt = format_prompt(conversation, next_speaker)
    response = generate_response(prompt)
    response_line = f"{next_speaker}: {response}"
    print(response_line)
    conversation.append(response_line)
    current_speaker = next_speaker