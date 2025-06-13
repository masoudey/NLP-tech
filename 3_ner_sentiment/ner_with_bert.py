# ner_with_bert.py
# Detailed implementation for Named Entity Recognition (NER) using a BERT-based model
# from Hugging Face, fine-tuned on a NER dataset (e.g., CoNLL-2003). No use of pipeline.
# Includes training (skipped if already done), evaluation, prediction on custom sentences
# (including user‐input extras), saving metrics to CSV, and in-script visualizations.

import os
import torch
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset
from evaluate import load as load_metric
import numpy as np

# Set device to CPU
device = torch.device("cpu")

# 1. Load the CoNLL-2003 NER dataset
ner_dataset = load_dataset("conll2003", trust_remote_code=True)
label_list = ner_dataset["train"].features["ner_tags"].feature.names
num_labels = len(label_list)

# 2. Load the BERT tokenizer and model
model_checkpoint = "bert-base-cased"
tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint)
model = BertForTokenClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

# 3. Tokenize the dataset
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                # For subwords, set label to -100 so they're ignored in loss
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = ner_dataset.map(tokenize_and_align_labels, batched=True)

# 4. Define metrics for evaluation
metric = load_metric("seqeval")
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall":    results["overall_recall"],
        "f1":        results["overall_f1"],
        "accuracy":  results["overall_accuracy"],
    }

# 5. Training arguments
training_args = TrainingArguments(
    output_dir="./ner_bert_results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,  # For demonstration, set to 1. Increase for real training.
    weight_decay=0.01,
    logging_dir="./ner_bert_logs",
    logging_steps=10,
    save_strategy="epoch",
    report_to="none"
)

data_collator = DataCollatorForTokenClassification(tokenizer)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 7. Train the model (skip if checkpoint exists)
last_checkpoint = get_last_checkpoint(training_args.output_dir)
if last_checkpoint:
    print(f"Found existing checkpoint at {last_checkpoint}, loading model instead of training.")
    model = BertForTokenClassification.from_pretrained(last_checkpoint, num_labels=num_labels)
    trainer.model = model
else:
    trainer.train()
    trainer.save_model()

# 8. Evaluate the model
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

# --- Export NER metrics to CSV ---
import pandas as pd
ner_metrics_df = pd.DataFrame([{
    "precision": eval_results["eval_precision"],
    "recall":    eval_results["eval_recall"],
    "f1":        eval_results["eval_f1"],
    "accuracy":  eval_results["eval_accuracy"]
}])
ner_metrics_df.to_csv("ner_metrics.csv", index=False)
print("Saved evaluation metrics to ner_metrics.csv")

# 9. Base custom sentences
custom_sentences = [
    "Apple is looking at buying U.K. startup for $1 billion.",
    "Barack Obama was born in Hawaii.",
    "Google was founded in September 1998.",
    "Elon Musk is the CEO of SpaceX.",
    "Amazon is based in Seattle, Washington.",
    "The Eiffel Tower is in Paris.",
    "Cristiano Ronaldo plays for Al Nassr.",
    "The United Nations is headquartered in New York.",
    "Mount Everest is the highest mountain in the world.",
    "The Mona Lisa is displayed in the Louvre Museum."
]

# 10. Prompt user for extra sentences
print("Enter any additional sentences (one per line). Leave blank and press Enter to finish:")
while True:
    line = input().strip()
    if not line:
        break
    custom_sentences.append(line)

# 11. Predict on all sentences
model.eval()
model = model.to(device)

ner_pred_rows = []
for sent in custom_sentences:
    tokens = tokenizer(sent, return_tensors="pt")
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        output = model(**tokens)
    predictions = torch.argmax(output.logits, dim=2)[0].cpu().numpy()
    tokens_decoded = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
    pred_labels = [label_list[p] for p in predictions]

    print(f"\nSentence: {sent}")
    print("Token\tLabel")
    for token, label in zip(tokens_decoded, pred_labels):
        print(f"{token}\t{label}")

    for token, label in zip(tokens_decoded, pred_labels):
        ner_pred_rows.append({
            "sentence":   sent,
            "token":      token,
            "true_label": "",
            "pred_label": label
        })

# —— Visualizations —— #
import matplotlib.pyplot as plt
from collections import Counter

# 1. Plot evaluation metrics
metrics = {
    "Precision": eval_results["eval_precision"],
    "Recall":    eval_results["eval_recall"],
    "F1":        eval_results["eval_f1"],
    "Accuracy":  eval_results["eval_accuracy"]
}
df_metrics = pd.Series(metrics)
plt.figure(figsize=(6, 4))
df_metrics.plot.bar()
plt.ylim(0, 1)
plt.title("NER Evaluation Metrics")
plt.ylabel("Score")
plt.tight_layout()
plt.show()

# 2. Display a table of sample predictions
df_preds = pd.DataFrame(ner_pred_rows)
print("\nSample NER predictions (first 10 rows):")
print(df_preds.head(10).to_string(index=False))

# 3. Plot distribution of predicted labels
pred_counter = Counter(df_preds["pred_label"])
labels, counts = zip(*pred_counter.items())
plt.figure(figsize=(8, 4))
plt.bar(labels, counts)
plt.xticks(rotation=90)
plt.title("NER Predicted-Label Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.show()