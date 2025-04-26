from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import torch
import joblib
import numpy as np
from transformers import pipeline
import re
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from camel_tools.utils.normalize import (
    normalize_unicode,
    normalize_alef_maksura_ar,
    normalize_alef_ar,
    normalize_teh_marbuta_ar
)
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.ner import NERecognizer
from camel_tools.utils.dediac import dediac_ar
from camel_tools.sentiment import SentimentAnalyzer

from nltk.corpus import stopwords
import nltk

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrendPredictor:
    def predict_trends(self, target_class: str, date: str):
        return np.random.rand() * 10


class TrendAnalyzer:
    def __init__(self):
        self.model_path = "./models"
        self.clf_model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.clf_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.label_encoder = joblib.load("label_encoder.pkl")
        self.trend_predictor = TrendPredictor()
        self.embed_model = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv2")
        self.embed_tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")

        # Load data with proper date parsing
        self.df = pd.read_csv("data/Data.csv")
        self.df["Date"] = pd.to_datetime(self.df["Date"], format='mixed', utc=True).dt.tz_localize(None)

        self.youtube_df = pd.read_csv("data/youtube_data.csv")
        self.youtube_df["Date"] = pd.to_datetime(self.youtube_df["Date"], format='mixed', utc=True).dt.tz_localize(None)

        self.df["predicted_label"] = self.df["Title"].apply(self.predict_class)
        self.df["source"] = "news"
        self.youtube_df["predicted_label"] = self.youtube_df["Title"].apply(self.predict_class)
        self.youtube_df["source"] = "youtube"

        nltk.download('stopwords')  # Only needed once
        self.arabic_stopwords = set(stopwords.words('arabic'))

        # Add domain-specific stopwords (e.g., sports terms)
        self.domain_stopwords = {
            'كرة', 'مباراة', 'فريق', 'الدوري', 'الرياضية',
            'المنتخب', 'أبطال', 'مشاهدة', 'أخبار', 'معلق',
            'رياضة', 'هدف', 'حارس', 'ملعب', 'لاعب','دوري','مباراه','مشاهدة','كره','امام','موعد','قدم'

                                                                                                'ملخص', 'الجوله',
            'الانجليزي', 'اليوم', 'كامل',
            'العالم', 'شباب', 'افضل', 'ملحميه', 'التاريخيه',
            'اسيا', 'مصر', 'نصف', 'شاهد', 'سنه', 'سيناريو',
            'غريب', 'الاعلام', 'اياب', 'ممتعه', 'عندما', 'جيل',
        }
        self.all_stopwords = self.arabic_stopwords.union(self.domain_stopwords)

        try:
            self.disambiguator = MLEDisambiguator.pretrained()
            self.sentiment_analyzer = SentimentAnalyzer.pretrained()
            self.ner = NERecognizer.pretrained()
            self.camel_available = True
        except Exception as e:
            print(f"CAMeL Tools initialization failed: {str(e)}")
            self.camel_available = False

    def clean_text(self, text: str) -> List[str]:
        if not text or not isinstance(text, str):
            return []

        try:
            # Step 1: Comprehensive Normalization
            text = normalize_unicode(text)
            text = normalize_alef_ar(text)  # Normalize all Alef variants
            text = normalize_alef_maksura_ar(text)  # Normalize ى to ي
            text = normalize_teh_marbuta_ar(text)  # Normalize ة to ه
            text = dediac_ar(text)  # Remove diacritics

            # Step 2: Advanced Cleaning
            text = re.sub(r'[^\u0621-\u064A0-9\s]', '', text)  # Keep only Arabic letters and numbers
            text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace

            # Step 3: Tokenization with CAMeL Tools
            tokens = simple_word_tokenize(text)

            # Step 4: Smart Filtering
            return [
                token for token in tokens
                if (len(token) > 2 and  # Minimum length requirement
                    token not in self.all_stopwords and  # Combined stopwords (NLTK + custom)
                    not token.isdigit() and  # Remove pure numbers
                    not re.match(r'^[\u0621-\u064A]{1,2}$', token)  # Remove 1-2 letter Arabic words
                    )
            ]
        except Exception as e:
            print(f"Text cleaning error: {str(e)}")
            return []

    def predict_class(self, text: str) -> int:
        inputs = self.clf_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.clf_model(**inputs)
        return torch.argmax(torch.nn.functional.softmax(outputs.logits, dim=1)).item()

    def get_embedding(self, text: str) -> np.ndarray:
        inputs = self.embed_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.embed_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

    def analyze_trends_from_df(self, df: pd.DataFrame, target_class: str, days: int) -> List[Dict]:
        try:
            encoded_label = self.label_encoder.transform([target_class])[0]
        except ValueError:
            return []

        cutoff_date = datetime.now() - timedelta(days=days)
        df_recent = df[(df["Date"] >= cutoff_date) & (df["predicted_label"] == encoded_label)]

        all_tokens = []
        for title in df_recent["Title"]:
            all_tokens.extend(self.clean_text(title))
        word_freq = Counter(all_tokens)

        keyword_vector = self.get_embedding(target_class)
        trends = []

        for word, freq in word_freq.most_common(100):
            if word == target_class:
                continue

            word_vector = self.get_embedding(word)
            sim = cosine_similarity([keyword_vector], [word_vector])[0][0]
            trends.append({
                "term": word,
                "score": float(sim * freq),
                "frequency": freq,
                "similarity": float(sim)
            })

        return sorted(trends, key=lambda x: x["score"], reverse=True)

    def normalize_youtube_metrics(self, df):
        df = df.copy()
        for metric in ['view_count', 'like_count', 'comment_count']:
            if metric in df.columns:
                max_val = df[metric].max()
                if max_val > 0:
                    df[f'normalized_{metric}'] = df[metric] / max_val
                else:
                    df[f'normalized_{metric}'] = 0
        return df

    def calculate_engagement_score(self, row):
        weights = {'view_count': 0.5, 'like_count': 0.3, 'comment_count': 0.2}
        score = 0
        for metric, weight in weights.items():
            if f'normalized_{metric}' in row:
                score += weight * row[f'normalized_{metric}']
        return score

    def analyze_youtube_trends(self, target_class: str, days: int) -> List[Dict]:
        try:
            encoded_label = self.label_encoder.transform([target_class])[0]
        except ValueError:
            return []

        cutoff_date = datetime.now() - timedelta(days=days)
        df_recent = self.youtube_df[
            (self.youtube_df["Date"] >= cutoff_date) &
            (self.youtube_df["predicted_label"] == encoded_label)
            ].copy()

        if df_recent.empty:
            return []

        df_recent = self.normalize_youtube_metrics(df_recent)
        df_recent['engagement_score'] = df_recent.apply(self.calculate_engagement_score, axis=1)

        # Create frequency and engagement dictionaries
        term_engagement = defaultdict(float)
        term_frequency = defaultdict(int)

        for title, engagement in zip(df_recent["Title"], df_recent["engagement_score"]):
            terms = self.clean_text(title)
            for term in terms:
                term_engagement[term] += engagement
                term_frequency[term] += 1

        # Convert to list of dictionaries with only engagement and frequency
        youtube_trends = []
        for term in term_frequency:
            youtube_trends.append({
                'term': term,
                'engagement_score': term_engagement[term],
                'frequency': term_frequency[term],
                'score': term_engagement[term]
            })

        return sorted(youtube_trends, key=lambda x: x['engagement_score'], reverse=True)

    def combine_trends(self, news_trends: List[Dict], youtube_trends: List[Dict]) -> tuple:
        news_dict = {item['term']: item for item in news_trends}
        youtube_dict = {item['term']: item for item in youtube_trends}

        combined = []
        news_only = []
        youtube_only = []

        all_terms = set(news_dict.keys()).union(set(youtube_dict.keys()))

        for term in all_terms:
            news_data = news_dict.get(term)
            youtube_data = youtube_dict.get(term)

            if news_data and youtube_data:
                youtube_score = youtube_data.get('engagement_score', youtube_data.get('score', 0))
                combined_score = (news_data['score'] * 0.5 + youtube_score * 0.5)
                combined.append({
                    'term': term,
                    'combined_score': combined_score,
                    'news_data': {
                        'score': news_data['score'],
                        'frequency': news_data['frequency']
                    },
                    'youtube_data': {
                        'frequency': youtube_data['frequency'],
                        'engagement': youtube_score,
                        'score': youtube_score
                    }
                })
            elif news_data:
                news_only.append(news_data)
            else:
                youtube_only.append(youtube_data)

        return (
            sorted(combined, key=lambda x: x['combined_score'], reverse=True),
            sorted(news_only, key=lambda x: x['score'], reverse=True),
            sorted(youtube_only, key=lambda x: x.get('engagement_score', x.get('score', 0)), reverse=True)
        )

    def get_time_series(self, target_class: str, days: int) -> Dict:
        encoded_label = self.label_encoder.transform([target_class])[0]
        cutoff_date = datetime.now() - timedelta(days=days)

        news_series = self.df[
            (self.df["predicted_label"] == encoded_label) &
            (self.df["Date"] >= cutoff_date)
            ].groupby(pd.Grouper(key="Date", freq="D")).size().reset_index(name="count")

        youtube_series = self.youtube_df[
            (self.youtube_df["predicted_label"] == encoded_label) &
            (self.youtube_df["Date"] >= cutoff_date)
            ].groupby(pd.Grouper(key="Date", freq="D")).size().reset_index(name="count")

        return {
            "news": {
                "dates": news_series["Date"].dt.strftime("%Y-%m-%d").tolist(),
                "counts": news_series["count"].tolist()
            },
            "youtube": {
                "dates": youtube_series["Date"].dt.strftime("%Y-%m-%d").tolist(),
                "counts": youtube_series["count"].tolist()
            }
        }

    def extract_entities(self, texts: List[str]) -> Dict:
        if not self.camel_available:
            return {"by_type": {}, "co_occurrence": []}

        entities = defaultdict(Counter)
        all_entities = []

        for text in texts:
            if not text or not isinstance(text, str):
                continue

            try:
                normalized = normalize_unicode(text)
                tokens = simple_word_tokenize(normalized)
                preds = self.ner.predict_sentence(tokens)

                current_entity = []
                current_type = None

                for token, pred in zip(tokens, preds):
                    if pred.startswith('B-'):
                        if current_entity:
                            entity_text = ' '.join(current_entity)
                            if entity_text and current_type:
                                entities[current_type][entity_text] += 1
                                all_entities.append((current_type, entity_text))
                        current_entity = [token]
                        current_type = pred[2:]
                    elif pred.startswith('I-'):
                        current_entity.append(token)
                    else:
                        if current_entity:
                            entity_text = ' '.join(current_entity)
                            if entity_text and current_type:
                                entities[current_type][entity_text] += 1
                                all_entities.append((current_type, entity_text))
                        current_entity = []
                        current_type = None

                if current_entity:
                    entity_text = ' '.join(current_entity)
                    if entity_text and current_type:
                        entities[current_type][entity_text] += 1
                        all_entities.append((current_type, entity_text))

            except Exception as e:
                print(f"Error processing text for NER: {str(e)}")
                continue

        co_occur = defaultdict(int)
        doc_entities = defaultdict(set)

        for i, (type1, ent1) in enumerate(all_entities):
            doc_entities[i // 10].add(f"{type1}::{ent1}")

        for doc in doc_entities.values():
            entities_list = list(doc)
            for i in range(len(entities_list)):
                for j in range(i + 1, len(entities_list)):
                    type1, ent1 = entities_list[i].split("::")
                    type2, ent2 = entities_list[j].split("::")
                    if ent1 != ent2:
                        key = f"{type1}||{ent1}||{type2}||{ent2}"
                        co_occur[key] += 1

        serializable_co_occur = []
        for key, count in sorted(co_occur.items(), key=lambda x: -x[1])[:10]:
            parts = key.split("||")
            if len(parts) == 4:
                serializable_co_occur.append({
                    "type1": parts[0],
                    "entity1": parts[1],
                    "type2": parts[2],
                    "entity2": parts[3],
                    "count": count
                })

        return {
            "by_type": {k: dict(v.most_common(5)) for k, v in entities.items()},
            "co_occurrence": serializable_co_occur
        }

    def analyze_sentiment(self, texts: List[str]) -> Dict:
        if not self.sentiment_analyzer:
            return {"positive": 0, "negative": 0, "neutral": 0}

        counts = Counter()

        for text in texts:
            if not text or not isinstance(text, str):
                continue

            try:
                normalized = normalize_unicode(text)
                normalized = normalize_alef_maksura_ar(normalized)

                # Get confidence scores
                prediction = self.sentiment_analyzer.predict(normalized, return_confidence=True)
                label, confidence = prediction

                # Only accept prediction if confidence > threshold (e.g., 0.6)
                if confidence > 0.6:
                    counts[label] += 1
                else:
                    counts["neutral"] += 1

            except Exception as e:
                print(f"Sentiment error: {str(e)}")
                counts["neutral"] += 1

        return dict(counts)


    def calculate_trend_stats(self, trends):
        if not trends:
            return {}
        scores = [t['combined_score'] if 'combined_score' in t else t['score'] for t in trends]
        return {
            'count': len(trends),
            'avg_score': np.mean(scores),
            'max_score': max(scores),
            'min_score': min(scores)
        }


analyzer = TrendAnalyzer()


class AnalysisRequest(BaseModel):
    target_class: str
    days: int = 7


@app.post("/api/advanced_analysis")
async def advanced_analysis(request: AnalysisRequest):
    try:
        # Get trends from both sources
        news_trends = analyzer.analyze_trends_from_df(analyzer.df, request.target_class, request.days)
        youtube_trends = analyzer.analyze_youtube_trends(request.target_class, request.days)

        # Combine trends
        combined_trends, news_only_trends, youtube_only_trends = analyzer.combine_trends(news_trends, youtube_trends)

        # Get articles for advanced analysis
        encoded_label = analyzer.label_encoder.transform([request.target_class])[0]
        cutoff_date = datetime.now() - timedelta(days=request.days)

        news_articles = analyzer.df[
            (analyzer.df["predicted_label"] == encoded_label) &
            (analyzer.df["Date"] >= cutoff_date)
            ]["Title"].tolist()

        youtube_articles = analyzer.youtube_df[
            (analyzer.youtube_df["predicted_label"] == encoded_label) &
            (analyzer.youtube_df["Date"] >= cutoff_date)
            ]["Title"].tolist()

        all_articles = news_articles + youtube_articles

        return {
            "trends": {
                "combined": combined_trends[:100],
                "news_only": news_only_trends[:50],
                "youtube_only": youtube_only_trends[:50]
            },
            "analysis": {
                "sentiment": analyzer.analyze_sentiment(all_articles),
                "entities": analyzer.extract_entities(all_articles),
                "time_series": analyzer.get_time_series(request.target_class, request.days),
            },
            "stats": {
                "total_articles": len(news_articles) + len(youtube_articles),
                "news_articles": len(news_articles),
                "youtube_videos": len(youtube_articles),
                "trend_stats": {
                    "combined": analyzer.calculate_trend_stats(combined_trends),
                    "news_only": analyzer.calculate_trend_stats(news_only_trends),
                    "youtube_only": analyzer.calculate_trend_stats(youtube_only_trends)
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/top_categories")
async def get_top_categories():
    return {
        "categories": list(analyzer.label_encoder.classes_)
    }


@app.get("/api/check_topic/{topic}")
async def check_topic(topic: str):
    exists = topic in analyzer.label_encoder.classes_
    return {"exists": exists}


class ComparisonRequest(BaseModel):
    topics: List[str]
    days: int = 7


@app.post("/api/compare")
async def compare_topics(request: ComparisonRequest):
    results = {}
    for topic in request.topics:
        try:
            encoded_label = analyzer.label_encoder.transform([topic])[0]

            # Get trends from both sources
            news_trends = analyzer.analyze_trends_from_df(analyzer.df, topic, request.days)
            youtube_trends = analyzer.analyze_youtube_trends(topic, request.days)

            # Get relevant articles/videos
            cutoff_date = datetime.now() - timedelta(days=request.days)
            news_texts = analyzer.df[
                (analyzer.df["predicted_label"] == encoded_label) &
                (analyzer.df["Date"] >= cutoff_date)
                ]["Title"].tolist()

            youtube_texts = analyzer.youtube_df[
                (analyzer.youtube_df["predicted_label"] == encoded_label) &
                (analyzer.youtube_df["Date"] >= cutoff_date)
                ]["Title"].tolist()

            all_texts = news_texts + youtube_texts

            results[topic] = {
                "top_terms": [t["term"] for t in news_trends[:5]],
                "top_youtube_terms": [t["term"] for t in youtube_trends[:5]],
                "total_mentions": len(news_texts) + len(youtube_texts),
                "news_mentions": len(news_texts),
                "youtube_mentions": len(youtube_texts),
                "time_series": analyzer.get_time_series(topic, request.days),
                "sentiment": analyzer.analyze_sentiment(all_texts),
                "average_engagement": np.mean([t["engagement_score"] for t in youtube_trends]) if youtube_trends else 0
            }
        except ValueError:
            continue

    if len(results) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 valid topics to compare")

    return results

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)