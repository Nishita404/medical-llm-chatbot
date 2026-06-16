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

print("Loading fine-tuned adapter...")
ft_model = PeftModel.from_pretrained(base_model, "medical_llm_adapter")
ft_model.eval()

print("All models loaded!")

# -----------------------
# Answer function with memory
# -----------------------

def generate_response(model, history_tuples, question):
    prompt = ""
    for user_msg, bot_msg in history_tuples:
        prompt += f"### Instruction:\n{user_msg}\n\n### Response:\n{bot_msg}\n\n"
    prompt += f"### Instruction:\n{question}\n\n### Response:\n"

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


def chat(question, history, model_choice):
    if not question.strip():
        return "", history

    # Convert flat list to tuples for prompt building
    history_tuples = []
    for i in range(0, len(history) - 1, 2):
        if i + 1 < len(history):
            history_tuples.append((history[i], history[i+1]))

    model = ft_model if model_choice == "Fine-tuned model" else base_model
    response = generate_response(model, history_tuples, question)

    history.append(question)
    history.append(response)
    return "", history


def clear_chat():
    return [], []


def history_to_pairs(history):
    return [(history[i], history[i+1]) for i in range(0, len(history) - 1, 2)]


# -----------------------
# Gradio UI
# -----------------------

css = """
#sidebar { background: #1a1a2e; padding: 16px; border-radius: 12px; }
footer { display: none !important; }
"""

with gr.Blocks(title="Medical LLM Chatbot") as app:

    with gr.Row():

        # ----- Sidebar -----
        with gr.Column(scale=1, elem_id="sidebar"):
            gr.Markdown("## 🩺 MedChat")
            gr.Markdown("*Fine-tuned on MedQuAD*")

            model_choice = gr.Radio(
                choices=["Fine-tuned model", "Base model"],
                value="Fine-tuned model",
                label="Active model"
            )

            gr.Markdown("---")
            gr.Markdown("**Suggested questions**")

            q1 = gr.Button("💊 Symptoms of diabetes?", size="sm")
            q2 = gr.Button("❤️ How is hypertension treated?", size="sm")
            q3 = gr.Button("🩸 What causes anemia?", size="sm")
            q4 = gr.Button("🫁 How is pneumonia diagnosed?", size="sm")
            q5 = gr.Button("💊 Side effects of ibuprofen?", size="sm")

            gr.Markdown("---")
            clear_btn = gr.Button("🗑️ Clear chat", variant="secondary")

            gr.Markdown("---")
            gr.Markdown("*Not intended for real medical advice.*")

        # ----- Main chat area -----
        with gr.Column(scale=4):
            gr.Markdown("# 🩺 Medical LLM Chatbot")
            gr.Markdown("Ask any medical question. The AI remembers your conversation.")

            chatbot = gr.Chatbot(
                height=500,
                show_label=False,
            )

            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask a medical question...",
                    show_label=False,
                    scale=5,
                    container=False
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

    # ----- State -----
    history_state = gr.State([])

    # ----- Button actions -----
    send_btn.click(
        fn=chat,
        inputs=[msg_box, history_state, model_choice],
        outputs=[msg_box, history_state]
    ).then(
        fn=history_to_pairs,
        inputs=history_state,
        outputs=chatbot
    )

    msg_box.submit(
        fn=chat,
        inputs=[msg_box, history_state, model_choice],
        outputs=[msg_box, history_state]
    ).then(
        fn=history_to_pairs,
        inputs=history_state,
        outputs=chatbot
    )

    # Suggested question buttons
    for btn, question in [
        (q1, "What are the symptoms of diabetes?"),
        (q2, "How is hypertension treated?"),
        (q3, "What causes anemia?"),
        (q4, "How is pneumonia diagnosed?"),
        (q5, "What are the side effects of ibuprofen?")
    ]:
        btn.click(fn=lambda q=question: q, outputs=msg_box)

    # Clear chat
    clear_btn.click(
        fn=clear_chat,
        outputs=[history_state, chatbot]
    )

app.launch(
    share=False,
    theme=gr.themes.Soft(primary_hue="green"),
)