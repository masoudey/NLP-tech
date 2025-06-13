#!/usr/bin/env python3
# spam_detection_pipeline.py

import os
import re
import gc
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# PyTorch & Hugging Face
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
from datasets import Dataset, DatasetDict

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 0. Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device in use:", device)

# 1. NLTK setup (run once; comment out after first download)
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+|\S+@\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words]
    return ' '.join(tokens)

def main():
    # 2. Load local dataset
    data_file = os.path.join("data", "spam.csv")
    if not os.path.isfile(data_file):
        raise FileNotFoundError(f"No CSV file found at {data_file}")
    print("Loading dataset from:", data_file)
    df = pd.read_csv(data_file, encoding='latin-1')
    df = df.loc[:, ['v1', 'v2']]
    df.columns = ['label', 'text']
    print("Dataset loaded:", df.shape)
    print(df['label'].value_counts())

    # 3. Limit size
    n = min(50000, len(df))
    df = df.sample(n=n, random_state=42).reset_index(drop=True)
    print("Reduced dataset size:", df.shape)

    # 4. Map labels
    label_map = {'spam': 0, 'ham': 1}
    df['label_id'] = df['label'].map(label_map)
    print("Label distribution:\n", df['label'].value_counts())

    # 5. EDA plots
    plt.figure()
    df['label'].value_counts().plot(kind='bar')
    plt.title('Spam vs. Ham Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    plt.figure()
    df['length'] = df['text'].astype(str).str.len()
    plt.hist(df['length'], bins=50)
    plt.title('Email Length Distribution')
    plt.xlabel('Number of Characters')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    # 6. Clean text
    df['clean_text'] = df['text'].apply(clean_text)

    # 7. Top-20 tokens
    def basic_tokenize(s):
        return re.findall(r"\w+", s.lower())
    all_tokens = basic_tokenize(" ".join(df['clean_text']))
    common = Counter(all_tokens).most_common(20)
    tokens, counts = zip(*common)
    plt.figure()
    plt.bar(tokens, counts)
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 20 Tokens (Cleaned Emails)')
    plt.tight_layout()
    plt.show()

    # 8. Train/test split
    X = df['clean_text']
    y = df['label_id']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 9. Baseline TF–IDF + classifiers (cached)
    model_dir   = "models"
    tfidf_path  = os.path.join(model_dir, "tfidf.joblib")
    logreg_path = os.path.join(model_dir, "logreg.joblib")
    nb_path     = os.path.join(model_dir, "nb.joblib")
    os.makedirs(model_dir, exist_ok=True)

    if os.path.exists(tfidf_path) and os.path.exists(logreg_path) and os.path.exists(nb_path):
        print("Loading TF–IDF & baseline models from disk...")
        tfidf  = joblib.load(tfidf_path)
        logreg = joblib.load(logreg_path)
        nb     = joblib.load(nb_path)
    else:
        print("Training TF–IDF & baseline models...")
        tfidf  = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        logreg = LogisticRegression(max_iter=1000, random_state=42)
        nb     = MultinomialNB()
        tfidf.fit(pd.concat([X_train, X_test]))
        joblib.dump(tfidf, tfidf_path)
        logreg.fit(tfidf.transform(X_train), y_train)
        joblib.dump(logreg, logreg_path)
        nb.fit(tfidf.transform(X_train), y_train)
        joblib.dump(nb, nb_path)

    X_train_tfidf = tfidf.transform(X_train)
    X_test_tfidf  = tfidf.transform(X_test)
    models = {'LogisticRegression': logreg, 'MultinomialNB': nb}

    # 10. Evaluate & plot baselines
    baseline_results = {}
    for name, model in models.items():
        preds = model.predict(X_test_tfidf)
        baseline_results[name] = {
            'accuracy':  accuracy_score(y_test, preds),
            'precision': precision_score(y_test, preds, average='weighted'),
            'recall':    recall_score(y_test, preds, average='weighted'),
            'f1':        f1_score(y_test, preds, average='weighted')
        }
        print(f"{name} Classification Report:\n",
              classification_report(y_test, preds, target_names=list(label_map.keys())))

    baseline_df = pd.DataFrame(baseline_results).T
    baseline_df.plot(kind='bar')
    plt.title('Baseline Model Metrics')
    plt.ylabel('Score')
    plt.tight_layout()
    plt.show()

    # 11. Prepare Hugging Face datasets
    tok_datasets = DatasetDict({
        'train': Dataset.from_pandas(pd.DataFrame({
            'text': X_train.tolist(), 'label': y_train.tolist()
        })),
        'test':  Dataset.from_pandas(pd.DataFrame({
            'text': X_test.tolist(),  'label': y_test.tolist()
        }))
    })

    MODEL_NAME  = 'distilbert-base-uncased'
    results_dir = "results"
    ckpts = [
        d for d in os.listdir(results_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(results_dir, d))
    ]

    if ckpts:
        latest_ckpt = max(ckpts, key=lambda s: int(s.split("-")[1]))
        ckpt_path   = os.path.join(results_dir, latest_ckpt)
        print("Loading fine-tuned BERT from:", ckpt_path)
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        model     = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
    else:
        print("Fine-tuning DistilBERT...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model     = AutoModelForSequenceClassification.from_pretrained(
                        MODEL_NAME, num_labels=len(label_map)
                    )

    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length')

    tok_datasets = tok_datasets.map(tokenize_fn, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=results_dir,
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        seed=42
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_datasets['train'],
        eval_dataset=tok_datasets['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: {
            'accuracy':  accuracy_score(p.label_ids, p.predictions.argmax(-1)),
            'precision': precision_score(p.label_ids, p.predictions.argmax(-1), average='weighted'),
            'recall':    recall_score(p.label_ids, p.predictions.argmax(-1), average='weighted'),
            'f1':        f1_score(p.label_ids, p.predictions.argmax(-1), average='weighted')
        }
    )

    if not ckpts:
        print("Training DistilBERT...")
        # trainer.train()
        # trainer.save_model(results_dir)
        # tokenizer.save_pretrained(results_dir)

    eval_results = trainer.evaluate()
    print("DistilBERT evaluation:", eval_results)

    # 12. Final comparison (print)
    print("\n=== Final Comparison ===")
    print("LogReg:      ", baseline_results['LogisticRegression'])
    print("MultinomialNB:", baseline_results['MultinomialNB'])
    print("DistilBERT:  ", {
        'accuracy':  eval_results['eval_accuracy'],
        'precision': eval_results['eval_precision'],
        'recall':    eval_results['eval_recall'],
        'f1':        eval_results['eval_f1']
    })

    # 13. Plot all metrics
    final_df = baseline_df.copy()
    final_df.loc['DistilBERT'] = [
        eval_results['eval_accuracy'],
        eval_results['eval_precision'],
        eval_results['eval_recall'],
        eval_results['eval_f1']
    ]
    final_df.plot(kind='bar')
    plt.title('Model Comparison Metrics')
    plt.ylabel('Score')
    plt.tight_layout()
    plt.show()

    # 14. Classify custom email
    email = input("Enter an email to classify: ")
    cleaned_email = clean_text(email)
    # baseline
    # feats = tfidf.transform([cleaned_email])
    # pred_lr = logreg.predict(feats)[0]
    # pred_nb = nb.predict(feats)[0]
    # print(f"LogisticRegression says: {'spam' if pred_lr==0 else 'ham'}")
    # print(f"MultinomialNB says:       {'spam' if pred_nb==0 else 'ham'}")
    # BERT
    inputs = tokenizer(cleaned_email, truncation=True, padding='max_length', return_tensors='pt')
    inputs = {k: v.to(device) for k,v in inputs.items()}
    model.to(device)
    with torch.no_grad():
        bert_pred = model(**inputs).logits.argmax(dim=1).cpu().item()
    print(f"DistilBERT says:          {'spam' if bert_pred==0 else 'ham'}")

    # 15. Cleanup
    try:
        del X_train_tfidf, X_test_tfidf, tfidf
        del models, baseline_df, final_df
        df.drop(columns=['clean_text','length'], inplace=True)
        gc.collect()
    except NameError:
        pass

if __name__ == "__main__":
    main()