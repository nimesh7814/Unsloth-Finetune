# Fine-Tuning Large Language Models with Unsloth (Unsloth Version)

This project demonstrates the fine-tuning of Large Language Models (LLMs) using the **Unsloth** framework. This version is specifically optimized to use Unsloth's tools and features for efficient model loading, fine-tuning, and inference. The code is designed to leverage pre-quantized 4-bit models, enabling faster model loading and reducing memory usage while preventing out-of-memory (OOM) issues.

## Overview

The script provided in this repository focuses on the following steps:
- Loading pre-quantized models using the Unsloth framework.
- Preparing models for inference and fine-tuning.
- Fine-tuning models with a Q&A formatted dataset using Supervised Fine-Tuning (SFT) with LoRA adapters.
- Evaluating the model before and after training.
- Saving the fine-tuned model in multiple formats for various use cases.

## Key Features

- **Unsloth Optimization:** Utilizes Unsloth's framework for fine-tuning, benefiting from faster downloads, reduced memory consumption, and enhanced model performance.
- **Efficient Model Loading:** Uses 4-bit quantized models to speed up downloads and reduce memory usage.
- **Memory Management:** Clears GPU cache and displays GPU memory statistics to monitor resource usage.
- **Supervised Fine-Tuning:** Fine-tunes models using a dataset formatted for Q&A.
- **Low-Rank Adaptation (LoRA):** Enhances model training using LoRA adapters for improved parameter efficiency.
- **Flexible Storage:** Saves models in GGUF format with different quantization methods (f32, f16, q8_0).
- **Interactive Chat Interface:** Provides a CLI-based chat interface for interacting with the fine-tuned model.

## Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/nimesh7814/Unsloth-Finetune.git
   cd Unsloth-Finetune
