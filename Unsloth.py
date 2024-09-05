import torch
from transformers import TrainingArguments, Trainer

# Function to initialize the model
def initialize_model(model_name, max_seq_length, device, hf_token):
    from unsloth import FastLanguageModel
    print(f"\nInitializing model: {model_name}")
    model = FastLanguageModel.get_peft_model(model_name, hf_token=hf_token, device=device)
    tokenizer = model.tokenizer
    tokenizer.model_max_length = max_seq_length
    return model, tokenizer

# Function to show current GPU memory stats
def show_memory_stats():
    gpu_stats = torch.cuda.get_device_properties(0)
    reserved_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    total_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {total_memory} GB.")
    print(f"{reserved_memory} GB of memory reserved.")

# Function to fine-tune the model
def fine_tune_model(model, tokenizer, dataset, device, training_args):
    print("\nFine-tuning the model...")
    args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",
        eval_steps=training_args["eval_steps"],
        save_steps=training_args["save_steps"],
        learning_rate=training_args["learning_rate"],
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        max_steps=training_args["max_steps"],
        logging_dir="./logs"
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer
    )
    trainer.train()

# Function to save the model using selected quantization methods
def save_model_with_quantization(model, tokenizer, quant_methods, base_dir):
    for quant in quant_methods:
        quantized_dir = f"{base_dir}/{quant}"
        print(f"\nSaving model with {quant} quantization to {quantized_dir}...")
        model.save_pretrained(quantized_dir)
        tokenizer.save_pretrained(quantized_dir)

# Function to handle chat after training
def handle_chat_after_training(chat_choice, model, tokenizer, device):
    if chat_choice == "yes":
        def chat_with_model_transformers():
            print("\nChat with the model (type 'exit' to stop):")
            while True:
                prompt = input("You: ")
                if prompt.lower() == "exit":
                    break
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                outputs = model.generate(**inputs, max_length=512)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Model: {response}")
        chat_with_model_transformers()
