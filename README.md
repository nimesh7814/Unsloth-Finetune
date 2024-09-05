## Fine-Tuning Large Language Models with Unsloth (Unsloth Version)
This project demonstrates the fine-tuning of Large Language Models (LLMs) using the Unsloth framework. This version is specifically optimized to use Unsloth's tools and features for efficient model loading, fine-tuning, and inference. The code is designed to leverage pre-quantized 4-bit models, enabling faster model loading and reducing memory usage while preventing out-of-memory (OOM) issues.

## Overview
The script provided in this repository focuses on the following steps:

Loading pre-quantized models using the Unsloth framework.
Preparing models for inference and fine-tuning.
Fine-tuning models with a Q&A formatted dataset using Supervised Fine-Tuning (SFT) with LoRA adapters.
Evaluating the model before and after training.
Saving the fine-tuned model in multiple formats for various use cases.

## Key Features
Unsloth Optimization: Utilizes Unsloth's framework for fine-tuning, benefiting from faster downloads, reduced memory consumption, and enhanced model performance.
Efficient Model Loading: Uses 4-bit quantized models to speed up downloads and reduce memory usage.
Memory Management: Clears GPU cache and displays GPU memory statistics to monitor resource usage.
Supervised Fine-Tuning: Fine-tunes models using a dataset formatted for Q&A.
Low-Rank Adaptation (LoRA): Enhances model training using LoRA adapters for improved parameter efficiency.
Flexible Storage: Saves models in GGUF format with different quantization methods (f32, f16, q8_0).
Interactive Chat Interface: Provides a CLI-based chat interface for interacting with the fine-tuned model.

## Setup Instructions
Clone the Repository:

git clone https://github.com/nimesh7814/Unsloth-Finetune.git
cd Unsloth-Finetune
Use code with caution.

## Install Dependencies:
Install the required libraries using pip:

pip install -r requirements.txt
Use code with caution.

Set Hugging Face Token:

Before running the script, ensure you have set your Hugging Face access token:

## huggingface-cli login
Use code with caution.

Configure Run Settings:

In the run.py file, adjust the following settings:

Max Steps: max_steps=
Learning Rate: learning_rate=
Save Steps: save_steps=
Evaluation Steps: eval_steps=
Repository: Set the Hugging Face repository path to nimesh7814/NBRO-Chatbot-V1
Chat Option: Set chat_choice to yes or no based on whether you want to chat with the model after training.
Maximum Length: Set max_length to 2048

## Run the Script:
python run.py


## Supported Pre-Quantized Models
- unsloth/Meta-Llama-3.1-8B-bnb-4bit - Llama-3.1 15 trillion tokens model, 2x faster!
- unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit
- unsloth/Meta-Llama-3.1-70B-bnb-4bit
- unsloth/Meta-Llama-3.1-405B-bnb-4bit - 405 billion parameter model in 4-bit quantization.
- unsloth/Mistral-Nemo-Base-2407-bnb-4bit - New Mistral 12B model, 2x faster!
- unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit
- unsloth/mistral-7b-v0.3-bnb-4bit - Mistral v3 model, 2x faster!
- unsloth/mistral-7b-instruct-v0.3-bnb-4bit
- unsloth/Phi-3.5-mini-instruct - Phi-3.5 model, 2x faster!
- unsloth/Phi-3-medium-4k-instruct
- unsloth/gemma-2-9b-bnb-4bit
- unsloth/gemma-2-27b-bnb-4bit - Gemma model, 2x faster!

## License
This project is licensed under the MIT License.
