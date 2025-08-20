from datasets import Dataset
from transformers import AutoTokenizer
import pandas as pd
import wandb
import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support

model_name = "emilyalsentzer/Bio_ClinicalBERT"  # or use "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

df_filtered = pd.read_csv('filtered_evidences.csv')
label_list = df_filtered["label"].unique().tolist()

# Create a Hugging Face Dataset from the DataFrame
dataset = Dataset.from_pandas(df_filtered)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

tokenized_dataset = tokenized_dataset.rename_column("label_id", "labels")
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["test"]

num_labels = len(label_list)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    acc = (predictions == torch.tensor(labels)).float().mean().item()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=8,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=1000,
    logging_steps=100,
    save_steps=4000,
    logging_dir='./logs',
    report_to=["wandb"],
)

wandb.init(project="multiclass-classifier")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained("./multiclass_icd_finetuned")
tokenizer.save_pretrained("./multiclass_icd_finetuned")