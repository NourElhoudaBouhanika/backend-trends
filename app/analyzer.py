import pandas as pd
import torch
import joblib
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from datetime import datetime, timedelta
from collections import Counter
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class TrendAnalyzer:
    def __init__(self):
        self.model_path = "./models"
        self.clf_model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.clf_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.label_encoder = joblib.load("label_encoder.pkl")

        self.embed_model = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv2")
        self.embed_tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")

        self.df = pd.read_csv("data/Data.csv")
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.df["predicted_label"] = self.df["Title"].apply(self.predict_class)

        self.arabic_stopwords = set([
            'من', 'في', 'على', 'و', 'عن', 'إلى', 'أن', 'إن', 'كان', 'كما', 'هذا',
            'ضد', 'بعد', 'كرة', 'تؤكد', 'أمام', 'مباراة', 'هذه', 'ذلك', 'لكن'
        ])

    def clean_text(self, text):
        text = re.sub(r'[^\u0621-\u064A\s]', '', str(text))
        tokens = text.split()
        return [token for token in tokens
                if token not in self.arabic_stopwords and re.match(r'^[\u0621-\u064A]+$', token)]

    def predict_class(self, text):
        inputs = self.clf_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.clf_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        return torch.argmax(probs, dim=1).item()

    def get_embedding(self, text):
        inputs = self.embed_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.embed_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

    def analyze_trends(self, target_class: str, days: int = 7):
        try:
            encoded_label = self.label_encoder.transform([target_class])[0]
        except ValueError:
            return {"error": f"Class '{target_class}' not found"}

        cutoff_date = datetime.now() - timedelta(days=days)
        df_recent = self.df[
            (self.df["Date"] >= cutoff_date) &
            (self.df["predicted_label"] == encoded_label)
        ]

        all_tokens = []
        for title in df_recent["Title"]:
            all_tokens.extend(self.clean_text(title))

        word_freq = Counter(all_tokens)
        keyword_vector = self.get_embedding(target_class)
        results = []

        for word, freq in word_freq.items():
            word_vector = self.get_embedding(word)
            sim = cosine_similarity([keyword_vector], [word_vector])[0][0]
            results.append({
                "word": word,
                "score": float(sim * freq),
                "frequency": freq,
                "similarity": float(sim)
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)[:20]

    def get_top_words(self, days: int = 7):
        cutoff_date = datetime.now() - timedelta(days=days)
        df_recent = self.df[self.df["Date"] >= cutoff_date]

        all_tokens = []
        for title in df_recent["Title"]:
            all_tokens.extend(self.clean_text(title))

        word_freq = Counter(all_tokens)
        return word_freq.most_common(20)

    def similarity_between_words(self, word1: str, word2: str):
        try:
            vec1 = self.get_embedding(word1)
            vec2 = self.get_embedding(word2)
            similarity = cosine_similarity([vec1], [vec2])[0][0]
            return float(similarity)
        except Exception as e:
            return {"error": str(e)}

    def available_labels(self):
        return list(self.label_encoder.classes_)
