import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# -----------------------
# Load models
# -----------------------

model_name = "microsoft/Phi-3-mini-4k-instruct"

print("Loading tokenizer...")
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

print("Loading your fine-tuned adapter...")
ft_model = PeftModel.from_pretrained(base_model, "medical_llm_adapter")
ft_model.eval()

print("All models loaded! Starting app...")

# -----------------------
# Answer function
# -----------------------

def answer(question):
    if not question.strip():
        return "Please enter a medical question.", ""

    prompt = f"### Instruction:\n{question}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        # Base model answer
        base_out = base_model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        base_answer = tokenizer.decode(base_out[0], skip_special_tokens=True)
        base_answer = base_answer.split("### Response:")[-1].strip()

        # Fine-tuned model answer
        ft_out = ft_model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        ft_answer = tokenizer.decode(ft_out[0], skip_special_tokens=True)
        ft_answer = ft_answer.split("### Response:")[-1].strip()

    return base_answer, ft_answer

# -----------------------
# Gradio UI
# -----------------------

with gr.Blocks(title="Medical LLM Chatbot") as app:

    gr.Markdown("# Medical LLM Chatbot")
    gr.Markdown("### Base model vs Fine-tuned model — side by side comparison")
    gr.Markdown("*Fine-tuned on MedQuAD dataset using QLoRA on RTX 4060*")

    with gr.Row():
        question_box = gr.Textbox(
            label="Ask a medical question",
            placeholder="e.g. What are the symptoms of diabetes?",
            lines=2,
            scale=4
        )
        ask_btn = gr.Button("Ask", variant="primary", scale=1)

    gr.Markdown("### Suggested questions")
    with gr.Row():
        gr.Button("What are symptoms of diabetes?").click(
            fn=lambda: "What are symptoms of diabetes?",
            outputs=question_box
        )
        gr.Button("How is hypertension treated?").click(
            fn=lambda: "How is hypertension treated?",
            outputs=question_box
        )
        gr.Button("What causes anemia?").click(
            fn=lambda: "What causes anemia?",
            outputs=question_box
        )

    with gr.Row():
        base_output = gr.Textbox(
            label="Base model (Phi-3 original)",
            lines=10,
            interactive=False
        )
        ft_output = gr.Textbox(
            label="Your fine-tuned model (trained on MedQuAD)",
            lines=10,
            interactive=False
        )

    ask_btn.click(
        fn=answer,
        inputs=question_box,
        outputs=[base_output, ft_output]
    )

    question_box.submit(
        fn=answer,
        inputs=question_box,
        outputs=[base_output, ft_output]
    )

    gr.Markdown("---")
    gr.Markdown("*This is a demo project. Not intended for real medical advice.*")

app.launch(share=False)