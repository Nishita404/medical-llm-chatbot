import torch
import json
import os
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

app = Flask(__name__)

HISTORY_FILE = "chat_history.json"

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
# Chat history helpers
# -----------------------

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return {}
    with open(HISTORY_FILE, "r") as f:
        return json.load(f)

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

# -----------------------
# Generate response
# -----------------------

def generate_response(model, messages, question):
    prompt = ""
    for msg in messages:
        if msg["role"] == "user":
            prompt += f"### Instruction:\n{msg['content']}\n\n"
        else:
            prompt += f"### Response:\n{msg['content']}\n\n"
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

# -----------------------
# Routes
# -----------------------

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question", "").strip()
    session_id = data.get("session_id")
    model_choice = data.get("model_choice", "fine-tuned")

    if not question:
        return jsonify({"error": "Empty question"}), 400

    # Load history
    all_history = load_history()

    # Create new session if needed
    if not session_id or session_id not in all_history:
        session_id = str(uuid.uuid4())
        all_history[session_id] = {
            "title": question[:40],
            "created_at": datetime.now().isoformat(),
            "messages": []
        }

    messages = all_history[session_id]["messages"]

    # Pick model
    model = ft_model if model_choice == "fine-tuned" else base_model

    # Generate response
    response = generate_response(model, messages, question)

    # Save to history
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": response})
    all_history[session_id]["messages"] = messages
    save_history(all_history)

    return jsonify({
        "response": response,
        "session_id": session_id
    })

@app.route("/sessions", methods=["GET"])
def get_sessions():
    all_history = load_history()
    sessions = []
    for sid, data in all_history.items():
        sessions.append({
            "id": sid,
            "title": data.get("title", "Untitled"),
            "created_at": data.get("created_at", "")
        })
    # Most recent first
    sessions.sort(key=lambda x: x["created_at"], reverse=True)
    return jsonify(sessions)

@app.route("/session/<session_id>", methods=["GET"])
def get_session(session_id):
    all_history = load_history()
    if session_id not in all_history:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(all_history[session_id])

@app.route("/session/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    all_history = load_history()
    if session_id in all_history:
        del all_history[session_id]
        save_history(all_history)
    return jsonify({"success": True})

if __name__ == "__main__":
    app.run(debug=False, port=5000)