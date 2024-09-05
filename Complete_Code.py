import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# Print the version Bug fixes
print("Unsloth 2024.8 Appears to be working fine")

# Function to show current GPU memory stats
def show_memory_stats():
    gpu_stats = torch.cuda.get_device_properties(0)
    reserved_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    total_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {total_memory} GB.")
    print(f"{reserved_memory} GB of memory reserved.")

# Clear GPU memory
torch.cuda.empty_cache()

# Set the device to use GPU 0
device = torch.device("cuda:0")

# 4bit pre-quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit",
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",
    "unsloth/gemma-2-2b-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",
]

# Define parameters
max_seq_length = 2048  # Reduced sequence length for memory efficiency
dtype = None  # Auto-detect dtype
load_in_4bit = True

# Define directories
base_dir = "/home/nimesh/Chatbot/Unsloth"
pretrained_model_dir = os.path.join(base_dir, "PretrainedModel")
model_name = "unsloth/Phi-3-mini-4k-instruct"
model_path = os.path.join(pretrained_model_dir, model_name)
dataset_dir = os.path.join(base_dir, "Datasets")
trained_model_dir = os.path.join(base_dir, "TrainedModel")
lora_adapter_dir = os.path.join(trained_model_dir, "LoraAdapter")
merged_model_dir = os.path.join(trained_model_dir, "MergedModel")
vllm_dir = os.path.join(trained_model_dir, "VLLM")

# Create directories if they don't exist
for dir_path in [model_path, dataset_dir, trained_model_dir, lora_adapter_dir, merged_model_dir, vllm_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Hugging Face token
hf_token = "Enter You HF Token"

# Show initial GPU memory stats
print("\nInitial GPU Memory Stats:")
show_memory_stats()

# Download the model
print("\nDownloading the model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    cache_dir=model_path,
    token=hf_token,
    force_download=False
)

# Prepare the model for inference (this is typically called only once)
FastLanguageModel.for_inference(model)
print("Model downloaded and prepared for inference successfully.")

# Adding LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None
)

# Load the Q&A formatted dataset
dataset_path = "/home/nimesh/Chatbot/Unsloth/Datasets/nbro-alpaca-dataset/20240904/Training_Dataset.json"
print(f"\nLoading dataset from {dataset_path}...")
dataset = load_dataset("json", data_files=dataset_path, split="train")

# Function to format dataset into Q&A pairs
def format_qa_dataset(examples):
    # Use the correct column names based on your dataset
    questions = examples["Question"]  # Changed to 'Question'
    answers = examples["Answer"]      # Changed to 'Answer'
    texts = []
    for question, answer in zip(questions, answers):
        text = f"Q: {question}\nA: {answer}"
        texts.append(text)
    return {"text": texts}

# Apply the formatting function to the dataset
formatted_dataset = dataset.map(format_qa_dataset, batched=True)

print(f"\nFormatted dataset: {formatted_dataset}")

# Split the dataset into train and eval sets (90% train, 10% eval)
train_test_split = formatted_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Verify the dataset structure
print("\nTrain Dataset columns:", train_dataset.column_names)
print("Train Dataset:", train_dataset["text"][:1])

print("\nEval Dataset columns:", eval_dataset.column_names)
print("Eval Dataset:", eval_dataset["text"][:1])

# Function to generate a response (before training)
def chat_with_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Asking the question before training
question = "Explain the meaning of a landslide amber warning. My area has received a landslide amber warning. What does it mean?"
print("\nBefore Training...")
response_before = chat_with_model(question)
print(f"Response before training: {response_before}")

# Training the Model
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Add the evaluation dataset here
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=5000,
        learning_rate=1e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=trained_model_dir,
        hub_token=hf_token,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
    ),
)

trainer_stats = trainer.train()

# Show memory stats after training
print("\nGPU Memory Stats After Training:")
show_memory_stats()

# Evaluate the model after training
print("\nEvaluating the model after training...")
eval_results = trainer.evaluate()

# Print evaluation details
print("Evaluation Results:")
for key, value in eval_results.items():
    print(f"{key}: {value}")

# Save evaluation results to a file
with open(os.path.join(trained_model_dir, "Phi-3-mini-4k-instruct-results.txt"), "w") as file:
    for key, value in eval_results.items():
        file.write(f"{key}: {value}\n")

print("Evaluation results have been saved to 'evaluation_results.txt'.")

# Save the LoRA Adapters
model.save_pretrained(lora_adapter_dir)
tokenizer.save_pretrained(lora_adapter_dir)

# Save the merged model locally
model.save_pretrained_merged(merged_model_dir, tokenizer, save_method="merged_16bit")

# Save the model in GGUF format using different quantization methods
print("Saving the model in GGUF format...")

print("Saving in f32 quantization...")
model.save_pretrained_gguf("GGUF", tokenizer, quantization_method="f32")
print("Model saved in f32 quantization.")

print("Saving in f16 quantization...")
model.save_pretrained_gguf("GGUF", tokenizer, quantization_method="f16")
print("Model saved in f16 quantization.")

print("Saving in q8_0 quantization...")
model.save_pretrained_gguf("GGUF", tokenizer, quantization_method="q8_0")
print("Model saved in q8_0 quantization.")

# Save the merged model as VLLM
model.save_pretrained_merged(vllm_dir, tokenizer, save_method="merged_16bit")

# Reload the trained model for inference
print("Inference model and tokenizer saved successfully.")

# Reload the model for inference
print("\nReloading the trained model for inference...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=lora_adapter_dir,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    cache_dir=model_path,
    token=hf_token,
    force_download=False
)

# Prepare the reloaded model for inference
FastLanguageModel.for_inference(model)
print("Trained model loaded and prepared for inference successfully.")

# Function to generate a response (after reloading the trained model)
def chat_with_model_after_training(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Asking the question after training
print("\nAfter Training...")
response_after = chat_with_model_after_training(question)
print(f"Response after training: {response_after}")

# Start chat interface
def chat_with_model_transformers():
    while True:
        prompt = input("You: ")
        if prompt.lower() == "exit":
            break
        response = chat_with_model_after_training(prompt)
        print(f"Model: {response}")

# Uncomment this line to start the chat interface
chat_with_model_transformers()