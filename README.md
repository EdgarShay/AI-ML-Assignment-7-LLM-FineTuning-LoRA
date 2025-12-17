# AI-ML-Assignment-7-LLM-FineTuning-LoRA

## Written by Edgar  
## Course: Generative AI / Machine Learning  
## Assignment: Transfer Learning with PEFT (LoRA)

# Project Overview
This project demonstrates how to fine-tune a pre-trained Large Language Model (LLM) for a downstream text classification task using Parameter-Efficient Fine-Tuning (PEFT).  
Instead of updating all model weights, LoRA (Low-Rank Adaptation) is used to train only a small subset of parameters, reducing computational cost while maintaining performance.


# Task Description
## Task: Sentiment Analysis (Positive vs Negative)
## Dataset: IMDB Movie Reviews (Hugging Face Datasets)
 Base Model: distilbert-base-uncased
- Frameworks: Hugging Face Transformers, PEFT, PyTorch


## Dataset Preparation
- Loaded the IMDB dataset using Hugging Face Datasets
- Tokenized text using the model’s tokenizer
- Converted data to PyTorch tensors
- Created smaller train, validation, and test subsets for efficient training


## LoRA Configuration (PEFT)
The following LoRA parameters were used:

- (rank): 8  
- lora_alpha: 32  
- target_modules: q_lin, v_lin  
- lora_dropout: 0.1  
- Trainable Parameters: Only LoRA layers  
- Frozen Parameters: All base model weights  

This setup ensures that only a small fraction of the model is trained.

## Training Details
- Optimizer: AdamW (default via Trainer)
- Epochs: 3
- Batch Size: 8
- Learning Rate: 5e-5
- Training performed on CPU

##  Evaluation Results
Final evaluation on the test set:

- Accuracy: 0.85  
- F1-Score: 0.84  

- Exact values may vary slightly due to random initialization and subset sampling.

## Why LoRA is Parameter-Efficient
LoRA reduces computational cost by:
- Freezing all original model weights
- Injecting small, trainable low-rank matrices into attention layers
- Training only a tiny fraction of parameters

This makes fine-tuning feasible on limited hardware such as CPUs or low-memory GPUs.

## Example Predictions
- This movie was fantastic, I loved it! → Positive
- The movie was boring and too long. → Negative

## Repository Contents
- llm_lora_finetuning.ipynb – Full Jupyter Notebook
- requirements.txt – Required Python packages
- README.md` – Project documentation

##  Conclusion
This project demonstrates how PEFT techniques like LoRA enable efficient fine-tuning of large language models, 
making them practical even in resource-constrained environments.

