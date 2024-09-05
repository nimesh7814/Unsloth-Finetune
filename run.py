# User Settings (run.py)

import os
import torch
from Unsloth import (
    fine_tune_model,
    initialize_model,
    show_memory_stats,
    save_model_with_quantization,
    handle_chat_after_training
)
from datasets import load_dataset

# Minimum user settings

# 1. Model name (Choose from supported models, see Help.txt for details)
model_name = "unsloth/Phi-3-mini-4k-instruct"

# 2. Hugging Face token (required for accessing some models)
hf_token = "your_huggingface_token"

# 3. Hugging Face repository name for the dataset
dataset_repo = "nimesh7814/NBRO-Chatbot-V1"

# 4. Sequence length (adjust based on your model's capability)
max_seq_length = 2048

# 5. Quantization methods to use (See Help.txt for details)
selected_quant_methods = ["f32", "f16", "q8_0"]

# 6. Training settings
max_steps = 5000         # Set the number of training steps
learning_rate = 1e-4     # Set the learning rate
save_steps = 500         # Set how often to save checkpoints
eval_steps = 100         # Set how often to run evaluation

# Ask the user if they want to chat with the model after training
chat_choice = input("\nDo you want to chat with the model after training? (yes/no): ").strip().lower()

# Set the current working directory
base_dir = os.getcwd()

# Load dataset from Hugging Face
print(f"\nLoading the dataset from Hugging Face repository: {dataset_repo}...")
dataset = load_dataset(dataset_repo)

# Set the device to use GPU 0
device = torch.device("cuda:0")

# Initialize and fine-tune the model
print("\nInitial GPU Memory Stats:")
show_memory_stats()
torch.cuda.empty_cache()

# Initialize the model
model, tokenizer = initialize_model(
    model_name=model_name,
    max_seq_length=max_seq_length,
    device=device,
    hf_token=hf_token
)

# Training arguments based on user inputs
training_args = {
    "max_steps": max_steps,
    "learning_rate": learning_rate,
    "save_steps": save_steps,
    "eval_steps": eval_steps
}

# Fine-tune the model
fine_tune_model(model, tokenizer, dataset, device, training_args)

# Save the model using the selected quantization method
save_model_with_quantization(model, tokenizer, selected_quant_methods, base_dir)

# Handle chat based on user choice
handle_chat_after_training(chat_choice, model, tokenizer, device)
