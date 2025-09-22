# -----------------------------
# File: train_bart.py
# -----------------------------
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

# Load preprocessed CSV
df = pd.read_csv("./data/preprocessed/input.csv")

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

def preprocess(example):
    inputs = tokenizer(example["text"], truncation=True, padding="max_length", max_length=1024)
    labels = tokenizer(example.get("summary", ""), truncation=True, padding="max_length", max_length=142)
    inputs["labels"] = labels["input_ids"]
    return inputs

dataset = Dataset.from_pandas(df)
tokenized = dataset.map(preprocess, batched=True)

training_args = TrainingArguments(
    output_dir="./bart_model",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    save_steps=500,
)

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
trainer.train()
trainer.save_model("./bart_model")
print("Model trained and saved to ./bart_model")