import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def preprocess_data(df):
    # Rename columns to 'text' and 'label' expected by HuggingFace
    if "question" in df.columns:
        df["text"] = df["question"]
    elif "sentence" in df.columns:
        df["text"] = df["sentence"]

    if "answer" in df.columns:
        df["label"] = df["answer"].apply(lambda x: 1 if str(x).lower() in ["yes", "positive", "true"] else 0)
    elif "sentiment" in df.columns:
        df["label"] = df["sentiment"].apply(lambda x: 1 if str(x).lower() == "positive" else 0)

    return df[["text", "label"]]

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


