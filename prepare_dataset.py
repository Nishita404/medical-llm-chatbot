from datasets import load_dataset, Dataset

dataset = load_dataset("lavita/MedQuAD")
train_data = dataset["train"]

formatted_data = []

for example in train_data:

    instruction = example["question"]
    response = example["answer"]

    formatted_text = f"""### Instruction:
{instruction}

### Response:
{response}
"""

    formatted_data.append({"text": formatted_text})

# convert to huggingface dataset
formatted_dataset = Dataset.from_list(formatted_data)

print(formatted_dataset)
print("\nExample:")
print(formatted_dataset[0])