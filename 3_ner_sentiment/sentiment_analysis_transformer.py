# sentiment_analysis_transformer.py
# Detailed implementation for sentiment analysis using a transformer model from Hugging Face,
# fine-tuned on a real-world dataset (e.g., IMDb). No use of pipeline.
# Includes training (skipped if already done), evaluation (with multiple metrics),
# prediction (with extra user input), saving metrics to CSV, and in-script visualizations.

import os
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset
from evaluate import load as load_metric
import numpy as np

# Set device to CPU
device = torch.device("cpu")

# 1. Load the IMDb sentiment dataset
sentiment_dataset = load_dataset("imdb")
label_list = (
    sentiment_dataset["train"].features["label"].names
    if hasattr(sentiment_dataset["train"].features["label"], "names")
    else ["negative", "positive"]
)
num_labels = 2

# 2. Load the DistilBERT tokenizer and model
model_checkpoint = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_checkpoint)
model = DistilBertForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=num_labels
)

# 3. Tokenize the dataset
def preprocess_function(examples):
    return tokenizer(
        examples["text"], truncation=True, padding=True, max_length=256
    )

tokenized_datasets = sentiment_dataset.map(preprocess_function, batched=True)

# 4. Define metrics for evaluation (accuracy, precision, recall, f1)
metric_acc = load_metric("accuracy")
metric_prec = load_metric("precision")
metric_rec = load_metric("recall")
metric_f1 = load_metric("f1")

def compute_metrics(p):
    predictions, labels = p
    preds = np.argmax(predictions, axis=1)
    acc = metric_acc.compute(predictions=preds, references=labels)["accuracy"]
    prec = metric_prec.compute(predictions=preds, references=labels)["precision"]
    rec = metric_rec.compute(predictions=preds, references=labels)["recall"]
    f1  = metric_f1.compute(predictions=preds, references=labels)["f1"]
    return {
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1
    }

# 5. Training arguments
training_args = TrainingArguments(
    output_dir="./sentiment_transformer_results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,  # For demonstration, set to 1. Increase for real training.
    weight_decay=0.01,
    logging_dir="./sentiment_transformer_logs",
    logging_steps=10,
    save_strategy="epoch",
    report_to="none"
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"]
        .shuffle(seed=42)
        .select(range(2000)),  # Use a subset for demo
    eval_dataset=tokenized_datasets["test"]
        .shuffle(seed=42)
        .select(range(500)),  # Use a subset for demo
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 7. Train the model (skip if checkpoint exists)
last_checkpoint = get_last_checkpoint(training_args.output_dir)
if last_checkpoint:
    print(f"Found existing checkpoint at {last_checkpoint}, loading model instead of training.")
    model = DistilBertForSequenceClassification.from_pretrained(
        last_checkpoint, num_labels=num_labels
    )
    trainer.model = model
else:
    trainer.train()
    trainer.save_model()

# 8. Evaluate the model
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

# --- Export Sentiment metrics to CSV ---
import pandas as pd
sent_metrics_df = pd.DataFrame([{
    "accuracy":  eval_results["eval_accuracy"],
    "precision": eval_results["eval_precision"],
    "recall":    eval_results["eval_recall"],
    "f1":        eval_results["eval_f1"]
}])
sent_metrics_df.to_csv("sentiment_metrics.csv", index=False)
print("Saved evaluation metrics to sentiment_metrics.csv")

# 9. Sample predictions
examples = [
    "I absolutely loved this movie! The acting was fantastic.",
    "This was the worst film I have ever seen.",
    "The plot was interesting but the pacing was slow.",
    "Great direction and wonderful performances.",
    "I wouldn't recommend this movie to anyone.",
    "A masterpiece of modern cinema.",
    "The story was predictable and boring.",
    "I enjoyed every minute of it!",
    "Terrible script and bad acting.",
    "An emotional and powerful experience."
]

# 10. Prompt user for extra texts
print("Enter any additional texts for prediction (one per line). Leave blank and press Enter to finish:")
while True:
    line = input().strip()
    if not line:
        break
    examples.append(line)

# 11. Predict on all texts
model.eval()
model = model.to(device)

sent_pred_rows = []
for text in examples:
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=256
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    pred_id = torch.argmax(outputs.logits, dim=1).item()
    label = label_list[pred_id]
    print(f"\nText: {text}\nPredicted Sentiment: {label}\n")

    sent_pred_rows.append({
        "text":       text,
        "true_label": "",
        "pred_label": label
    })

# —— Visualizations —— #
import matplotlib.pyplot as plt
from collections import Counter

# 1. Plot all evaluation metrics
metrics = {
    "Accuracy":  eval_results["eval_accuracy"],
    "Precision": eval_results["eval_precision"],
    "Recall":    eval_results["eval_recall"],
    "F1":        eval_results["eval_f1"]
}
df_metrics = pd.Series(metrics)
plt.figure(figsize=(6, 4))
df_metrics.plot.bar()
plt.ylim(0, 1)
plt.title("Sentiment Analysis Evaluation Metrics")
plt.ylabel("Score")
plt.tight_layout()
plt.show()

# 2. Display a table of sample predictions
df_sent = pd.DataFrame(sent_pred_rows)
print("\nSample sentiment predictions (first 10 rows):")
print(df_sent.head(10).to_string(index=False))

# 3. Plot distribution of predicted labels
dist = Counter(df_sent["pred_label"])
labels, counts = zip(*dist.items())
plt.figure(figsize=(4, 4))
plt.bar(labels, counts)
plt.title("Sentiment Prediction Counts")
plt.tight_layout()
plt.show()