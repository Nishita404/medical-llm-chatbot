from datasets import load_dataset

dataset = load_dataset("lavita/MedQuAD")

print(dataset)

print("\nColumns:")
print(dataset["train"].column_names)

print("\nFirst example:")
print(dataset["train"][0])