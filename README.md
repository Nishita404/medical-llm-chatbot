# Medical LLM Chatbot

A medical question-answering chatbot built by fine-tuning an open-source LLM on the MedQuAD dataset.

## Highlights

- Fine-tuned an open-source LLM on the MedQuAD medical dataset
- Implemented QLoRA parameter-efficient fine-tuning
- Built a Gradio interface for interactive medical question answering
- Compared base model vs fine-tuned model responses

## Features
- Fine-tuned LLM using QLoRA
- Medical question answering
- Gradio interface
- Base vs Fine-tuned model comparison

## Tech Stack
- Python
- HuggingFace Transformers
- PEFT (QLoRA)
- PyTorch
- Gradio

## Project Structure

medical-llm-chatbot
│
├── app.py                # Gradio interface
├── train_model.py        # Fine-tuning script
├── prepare_dataset.py    # Dataset preprocessing
├── explore_dataset.py    # Dataset exploration
├── test_model.py         # Model testing
└── test_tinyllama.py     # Base model testing

## Run Locally

Install dependencies:

pip install -r requirements.txt

Run the app:

python app.py

## Disclaimer
This project is for educational purposes and not intended for real medical advice.
