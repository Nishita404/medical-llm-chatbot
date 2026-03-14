import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# -----------------------
# Load base model
# -----------------------

model_name = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# -----------------------
# Load your fine-tuned adapter
# -----------------------

print("Loading your fine-tuned adapter...")
ft_model = PeftModel.from_pretrained(base_model, "medical_llm_adapter")
ft_model.eval()

# -----------------------
# Test function
# -----------------------

def ask(model, question):
    prompt = f"### Instruction:\n{question}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:")[-1].strip()

# -----------------------
# Compare base vs fine-tuned
# -----------------------

questions = [
    "What are the symptoms of diabetes?",
    "How is hypertension treated?",
    "What causes anemia?",
    "What are the side effects of ibuprofen?",
    "How is pneumonia diagnosed?"
]

print("\n" + "="*60)
print("BASE MODEL vs YOUR FINE-TUNED MODEL")
print("="*60)

for q in questions:
    print(f"\nQuestion: {q}")
    print("-" * 40)
    print("Base model:", ask(base_model, q))
    print()
    print("Fine-tuned:", ask(ft_model, q))
    print("="*60)