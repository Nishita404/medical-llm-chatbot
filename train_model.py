import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer

# -----------------------
# Load dataset
# -----------------------

dataset = load_dataset("lavita/MedQuAD")
train_data = dataset["train"]

formatted_data = []

for example in train_data.select(range(2000)):
    formatted_data.append({
        "text": f"### Instruction:\n{example['question']}\n\n### Response:\n{example['answer']}"
    })

train_dataset = Dataset.from_list(formatted_data)

print("Dataset loaded")
print(train_dataset)


# -----------------------
# Model
# -----------------------

model_name = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# 4bit quantization (QLoRA) — use bfloat16 for Phi-3
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16   # FIXED
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16              # FIXED
)

model.config.use_cache = False


# -----------------------
# LoRA config
# -----------------------

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["qkv_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


# -----------------------
# Training arguments
# -----------------------

training_args = TrainingArguments(
    output_dir="./medical_llm",

    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,

    num_train_epochs=1,

    logging_steps=10,
    save_steps=100,

    learning_rate=2e-4,

    fp16=False,      # FIXED
    bf16=True,       # FIXED — Phi-3 needs bfloat16

    optim="adamw_torch",

    report_to="none"
)


# -----------------------
# Trainer
# -----------------------

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    args=training_args
)


# -----------------------
# Train
# -----------------------

trainer.train()


# -----------------------
# Save adapter
# -----------------------

trainer.save_model("medical_llm_adapter")

print("Training finished successfully")