from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
import pandas as pd
import torch
import joblib
import numpy as np
import re
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import json
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

genai.configure(api_key="AIzaSyD77B4XJnspFZpRMeYqFZX7JEQMUWMTxpo")

############################################ Trends Extraction + text preprocessing ##########################################
class TrendAnalyzer:
    def __init__(self):
        self.model_path = "./models"
        self.clf_model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.clf_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.label_encoder = joblib.load("label_encoder.pkl")
        self.embed_model = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv2")
        self.embed_tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")

        self.df = pd.read_csv("data/Data.csv")
        self.df["Date"] = pd.to_datetime(self.df["Date"], format='mixed', utc=True).dt.tz_localize(None)

        self.youtube_df = pd.read_csv("data/youtube_data.csv")
        self.youtube_df["Date"] = pd.to_datetime(self.youtube_df["Date"], format='mixed', utc=True).dt.tz_localize(None)

        self.df["predicted_label"] = self.df["Title"].apply(self.predict_class)
        self.df["source"] = "news"
        self.youtube_df["predicted_label"] = self.youtube_df["Title"].apply(self.predict_class)
        self.youtube_df["source"] = "youtube"

        nltk.download('stopwords')
        self.arabic_stopwords = set(stopwords.words('arabic'))
        self.domain_stopwords = {
            'كرة', 'مباراة', 'فريق', 'الدوري', 'الرياضية',
            'المنتخب', 'أبطال', 'مشاهدة', 'أخبار', 'معلق',
            'رياضة', 'هدف', 'حارس', 'ملعب', 'لاعب','دوري','مباراه','مشاهدة',
            'كره','امام','موعد','قدم','ملخص','الجوله',
            'الانجليزي', 'اليوم', 'كامل',
            'العالم', 'شباب', 'افضل', 'ملحميه', 'التاريخيه',
            'اسيا', 'مصر', 'نصف', 'شاهد', 'سنه', 'سيناريو',
            'غريب', 'الاعلام', 'اياب', 'ممتعه', 'عندما', 'جيل',
        }
        self.all_stopwords = self.arabic_stopwords.union(self.domain_stopwords)

        try:
            self.disambiguator = MLEDisambiguator.pretrained()
            self.ner = NERecognizer.pretrained()
            self.camel_available = True
        except Exception as e:
            print(f"CAMeL Tools initialization failed: {str(e)}")
            self.camel_available = False

    def clean_text(self, text: str) -> List[str]:
        if not text or not isinstance(text, str):
            return []

        try:
            text = normalize_unicode(text)
            text = normalize_alef_ar(text)
            text = normalize_alef_maksura_ar(text)
            text = normalize_teh_marbuta_ar(text)
            text = dediac_ar(text)

            text = re.sub(r'[^\u0621-\u064A0-9\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()

            tokens = simple_word_tokenize(text)

            return [
                token for token in tokens
                if (len(token) > 2 and
                    token not in self.all_stopwords and
                    not token.isdigit() and
                    not re.match(r'^[\u0621-\u064A]{1,2}$', token)
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
        df_recent = self.youtube_df[(self.youtube_df["Date"] >= cutoff_date) & (self.youtube_df["predicted_label"] == encoded_label)].copy()

        if df_recent.empty:
            return []

        df_recent = self.normalize_youtube_metrics(df_recent)
        df_recent['engagement_score'] = df_recent.apply(self.calculate_engagement_score, axis=1)

        term_engagement = defaultdict(float)
        term_frequency = defaultdict(int)

        for title, engagement in zip(df_recent["Title"], df_recent["engagement_score"]):
            terms = self.clean_text(title)
            for term in terms:
                term_engagement[term] += engagement
                term_frequency[term] += 1

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

        news_series = self.df[(self.df["predicted_label"] == encoded_label) & (self.df["Date"] >= cutoff_date)].groupby(pd.Grouper(key="Date", freq="D")).size().reset_index(name="count")
        youtube_series = self.youtube_df[(self.youtube_df["predicted_label"] == encoded_label) & (self.youtube_df["Date"] >= cutoff_date)].groupby(pd.Grouper(key="Date", freq="D")).size().reset_index(name="count")

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
        pos_words = {
            'جيد', 'رائع', 'ممتاز', 'فوز', 'انتصار', 'نجاح', 'مبهر', 'مذهل', 'سعيد',
            'مفرح', 'مبهج', 'مبشر', 'تفوق', 'تميز', 'إنجاز', 'إبداع', 'احتراف', 'قوي',
            'متفوق', 'مبتكر', 'متميز', 'فريد', 'عظيم', 'رائعة', 'مدهش', 'مبهرة'
        }
        neg_words = {
            'سيء', 'خسارة', 'هزيمة', 'مشكلة', 'إصابة', 'فشل', 'خيبة', 'محزن', 'حزين',
            'مؤلم', 'مأساة', 'كارثة', 'انهيار', 'ضعف', 'تراجع', 'إهانة', 'خيانة', 'فوضى',
            'مخيب', 'مخزية', 'مأساوية', 'كارثية', 'مؤسفة', 'مخيبة', 'محبطة', 'مروعة'
        }

        counts = Counter()
        for text in texts:
            if not text:
                continue

            try:
                normalized = normalize_unicode(text)
                tokens = set(self.clean_text(normalized))
                pos = len(tokens & pos_words)
                neg = len(tokens & neg_words)

                if pos > neg:
                    counts["positive"] += 1
                elif neg > pos:
                    counts["negative"] += 1
                else:
                    counts["neutral"] += 1
            except Exception as e:
                print(f"Error analyzing sentiment: {str(e)}")
                continue

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
        news_trends = analyzer.analyze_trends_from_df(analyzer.df, request.target_class, request.days)
        youtube_trends = analyzer.analyze_youtube_trends(request.target_class, request.days)

        combined_trends, news_only_trends, youtube_only_trends = analyzer.combine_trends(news_trends, youtube_trends)

        encoded_label = analyzer.label_encoder.transform([request.target_class])[0]
        cutoff_date = datetime.now() - timedelta(days=request.days)

        news_articles = analyzer.df[(analyzer.df["predicted_label"] == encoded_label) & (analyzer.df["Date"] >= cutoff_date)]["Title"].tolist()
        youtube_articles = analyzer.youtube_df[(analyzer.youtube_df["predicted_label"] == encoded_label) & (analyzer.youtube_df["Date"] >= cutoff_date)]["Title"].tolist()
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


@app.get("/api/hero_trends")
async def get_hero_trends(days: int = 7, limit: int = 10):
    try:
        cutoff_date = datetime.now() - timedelta(days=days)

        combined_df = pd.concat([analyzer.df[["Title", "Date", "source"]],analyzer.youtube_df[["Title", "Date", "source"]]])
        recent_df = combined_df[combined_df["Date"] >= cutoff_date]

        all_terms = []
        for title in recent_df["Title"]:
            all_terms.extend(analyzer.clean_text(title))

        term_freq = Counter(all_terms)
        top_terms = term_freq.most_common(limit)

        return {
            "trends": [
                {"term": term, "count": count}
                for term, count in top_terms
            ],
            "time_period": f"Last {days} days"
        }

    except Exception as e:
        return {
            "error": str(e),
            "trends": [],
            "time_period": f"Last {days} days"
        }

@app.get("/api/top_categories")
async def get_top_categories():
    return {
        "categories": list(analyzer.label_encoder.classes_)
    }


@app.get("/api/check_topic/{topic}")
async def check_topic(topic: str):
    exists = topic in analyzer.label_encoder.classes_
    return {"exists": exists}


############################################ Comparing topics ##########################################
class ComparisonRequest(BaseModel):
    topics: List[str]
    days: int = 7

@app.post("/api/compare")
async def compare_topics(request: ComparisonRequest):
    results = {}
    for topic in request.topics:
        try:
            encoded_label = analyzer.label_encoder.transform([topic])[0]
            cutoff_date = datetime.now() - timedelta(days=request.days)

            news_trends = analyzer.analyze_trends_from_df(analyzer.df, topic, request.days)
            youtube_trends = analyzer.analyze_youtube_trends(topic, request.days)

            time_series = analyzer.get_time_series(topic, request.days)

            news_articles = analyzer.df[
                (analyzer.df["predicted_label"] == encoded_label) &
                (analyzer.df["Date"] >= cutoff_date)
            ]
            youtube_articles = analyzer.youtube_df[
                (analyzer.youtube_df["predicted_label"] == encoded_label) &
                (analyzer.youtube_df["Date"] >= cutoff_date)
            ]

            news_count = len(news_articles)
            youtube_count = len(youtube_articles)
            total_count = news_count + youtube_count

            all_texts = news_articles["Title"].tolist() + youtube_articles["Title"].tolist()
            sentiment = analyzer.analyze_sentiment(all_texts)

            avg_engagement = np.mean([t["engagement_score"] for t in youtube_trends]) if youtube_trends else 0

            results[topic] = {
                "trends": {
                    "combined": analyzer.combine_trends(news_trends, youtube_trends)[0][:5],
                    "news_only": news_trends[:5],
                    "youtube_only": youtube_trends[:5]
                },
                "time_series": time_series,
                "stats": {
                    "total_mentions": total_count,
                    "news_mentions": news_count,
                    "youtube_mentions": youtube_count,
                    "avg_engagement": float(avg_engagement)
                },
                "sentiment": sentiment,
                "entities": analyzer.extract_entities(all_texts)
            }
        except ValueError:
            continue

    if len(results) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 valid topics to compare")

    return {
        "comparison": results,
        "topics": request.topics,
        "time_period": f"Last {request.days} days"
    }

############################################ Search ##########################################
class SearchRequest(BaseModel):
    search_term: str
    days: Optional[int] = 30
    source: Optional[str] = None
    limit: Optional[int] = 20

@app.post("/api/search_articles")
async def search_articles(request: SearchRequest):
    try:
        is_label = request.search_term in analyzer.label_encoder.classes_

        cutoff_date = datetime.now() - timedelta(days=request.days)
        results = []

        dfs = []
        if request.source is None or request.source == "news":
            dfs.append(analyzer.df)
        if request.source is None or request.source == "youtube":
            dfs.append(analyzer.youtube_df)

        for df in dfs:
            if is_label:
                encoded_label = analyzer.label_encoder.transform([request.search_term])[0]
                filtered = df[(df["predicted_label"] == encoded_label) & (df["Date"] >= cutoff_date)]
            else:
                filtered = df[df["Title"].str.contains(request.search_term, case=False, na=False) & (df["Date"] >= cutoff_date)]

            for _, row in filtered.head(request.limit).iterrows():
                results.append({
                    "title": row["Title"],
                    "date": row["Date"].strftime("%Y-%m-%d") if pd.notna(row["Date"]) else None,
                    "source": row["source"],
                    "label": analyzer.label_encoder.inverse_transform([row["predicted_label"]])[
                        0] if "predicted_label" in row else None
                })

        return {
            "search_term": request.search_term,
            "is_label_search": is_label,
            "results": results[:request.limit],
            "total_results": len(results),
            "time_period": f"Last {request.days} days"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


############################################ Predictions Part ##########################################
class ConfidenceMetrics(BaseModel):
    self_consistency: float
    plausibility: float
    domain_relevance: float
    composite_confidence: float
    confidence_level: str

class PredictionResponse(BaseModel):
    prediction_id: str
    type: str
    keyword: str
    sport: Optional[str]
    prediction: str
    confidence: ConfidenceMetrics
    timestamp: str

class GeminiSportsPredictor:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.valid_sports = [
            'تنس', 'دراجات', 'سباحة', 'غولف', 'فورمولا 1',
            'كرة السلة', 'كرة الطائرة', 'كرة القدم', 'كريكيت', 'ملاكمة'
        ]
        self.consistency_cache = {}

    def _generate_content(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            if not response.text:
                raise ValueError("Empty response from Gemini API")
            return response.text
        except Exception as e:
            print(f"Generation error: {str(e)}")
            raise

    def _calculate_confidence(self, prompt: str, prediction: str) -> dict:
        try:
            consistency_score = self._check_self_consistency(prompt)
            plausibility_score = self._check_plausibility(prediction)
            domain_score = self._check_domain_relevance(prediction)
            confidence_score = 0.5 * consistency_score + 0.3 * plausibility_score + 0.2 * domain_score

            return {
                "self_consistency": consistency_score,
                "plausibility": plausibility_score,
                "domain_relevance": domain_score,
                "composite_confidence": confidence_score,
                "confidence_level": self._get_confidence_level(confidence_score)
            }
        except Exception as e:
            print(f"Confidence calculation error: {str(e)}")
            return {
                "self_consistency": 0,
                "plausibility": 0,
                "domain_relevance": 0,
                "composite_confidence": 0,
                "confidence_level": "غير معروف"
            }

    def _check_self_consistency(self, prompt: str, samples: int = 3) -> float:
        if prompt in self.consistency_cache:
            return self.consistency_cache[prompt]

        responses = []
        for _ in range(samples):
            response = self._generate_content(prompt)
            responses.append(response)

        similarities = []
        for a, b in combinations(responses, 2):
            emb_a = analyzer.get_embedding(a)
            emb_b = analyzer.get_embedding(b)
            sim = cosine_similarity([emb_a], [emb_b])[0][0]
            similarities.append(sim)

        avg_similarity = float(np.mean(similarities))
        self.consistency_cache[prompt] = avg_similarity
        return avg_similarity

    def _check_plausibility(self, text: str) -> float:
        indicators = [
            r'توقع', r'نتيجة', r'فريق', r'لاعب', r'مباراة',
            r'بطولة', r'هدف', r'فوز', r'خسارة', r'نسبة',
            r'\d+%', r'أسباب', r'عوامل', r'تحليل'
        ]
        matches = sum(1 for pattern in indicators if re.search(pattern, text))
        return min(1.0, matches / 10)

    def _check_domain_relevance(self, text: str) -> float:
        sports_terms = [
            'رياضة', 'رياضي', 'ملعب', 'دوري', 'كأس',
            'مباراة', 'تسديد', 'حارس', 'هدف', 'بطولة'
        ]
        matches = sum(1 for term in sports_terms if term in text)
        return min(1.0, matches / 5)

    def _get_confidence_level(self, score: float) -> str:
        if score >= 0.8:
            return "عالية جدًا"
        elif score >= 0.6:
            return "عالية"
        elif score >= 0.4:
            return "متوسطة"
        else:
            return "منخفضة"
    def predict(self, prediction_type: str, keyword: str, sport: Optional[str] = None) -> dict:
        """Synchronous prediction method"""
        try:
            if prediction_type == "trend":
                if not sport:
                    raise ValueError("Sport is required for trend predictions")
                prompt = self._create_future_trends_prompt(sport)
            else:
                prompt = self._create_match_prediction_prompt(keyword)

            prediction_text = self._generate_content(prompt)
            confidence = self._calculate_confidence(prompt, prediction_text)

            return {
                "prediction_id": str(uuid.uuid4()),
                "type": prediction_type,
                "keyword": keyword,
                "sport": sport,
                "prediction": prediction_text,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "timeframe": "7 أيام قادمة" if prediction_type == "trend" else "المباراة القادمة"
            }
        except Exception as e:
            error_msg = f"Error in prediction: {str(e)}"
            print(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

    def _create_future_trends_prompt(self, sport: str) -> str:
        """Generate prompt specifically for future trends prediction"""
        return f"""
        أنت خبير في تحليل اتجاهات الرياضة العربية. الرياضة المحددة هي: {sport}

        المطلوب:
        1. توقع أهم 5 اتجاهات أو مواضيع متعلقة بهذه الرياضة في الأيام 7 القادمة (وليس الحالية)
        2. لكل اتجاه، اذكر:
           - سبب ظهوره المتوقع في الأسبوع القادم
           - اللاعبين/الفرق/الأحداث المرتبطة به
           - مدى تأثيره على المشهد الرياضي
           - احتمالية حدوثه (من 0% إلى 100%)
        3. رتب الاتجاهات حسب:
           - الأكثر احتمالاً للحدوث
           - الأكثر تأثيراً على الرياضة

        ملاحظات مهمة:
        - ركز فقط على ما سيحدث في الأيام 7 القادمة
        - لا تذكر أي أحداث حالية
        - كن دقيقاً في التواريخ المتوقعة

        أجب باللغة العربية وبشكل منظم مع عناوين واضحة.
        """

    def _create_match_prediction_prompt(self, teams: str) -> str:
        """Generate prompt for match predictions"""
        return f"""
        أنت محلل رياضي محترف. المطلوب تحليل المباراة بين: {teams}

        أجب باللغة العربية وبشكل منظم:
        1. نظرة عامة على الفرق/اللاعبين
        2. المقارنة الفنية (نقاط القوة والضعف)
        3. العوامل المؤثرة (إصابات، ظروف، إلخ)
        4. التوقع النهائي مع:
           - النتيجة المتوقعة
           - النسبة المئوية للفوز
           - أهم اللاعبين الذين سيؤثرون في المباراة
        5. التوقعات طويلة المدى بعد هذه المباراة

        قدم إجابة مفصلة ومنظمة مع التركيز على التحليل المستقبلي.
        """




class AccuracyEvaluator:
    def __init__(self):
        self.predictions_log = "predictions_log.json"
        self.accuracy_log = "accuracy_log.json"

    def _load_log(self, log_file: str) -> list:
        if not os.path.exists(log_file):
            return []
        with open(log_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_log(self, log_file: str, data: list):
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def log_prediction(self, prediction: dict):
        log = self._load_log(self.predictions_log)
        log.append(prediction)
        self._save_log(self.predictions_log, log)

    def record_accuracy(self, prediction_id: str, actual_outcome: str, similarity: float):
        accuracy_log = self._load_log(self.accuracy_log)
        accuracy_log.append({
            "prediction_id": prediction_id,
            "actual_outcome": actual_outcome,
            "similarity": similarity,
            "evaluation_date": datetime.now().isoformat()
        })
        self._save_log(self.accuracy_log, accuracy_log)

    def get_accuracy_stats(self) -> dict:
        accuracy_log = self._load_log(self.accuracy_log)
        if not accuracy_log:
            return {"message": "No accuracy data available"}

        similarities = [entry["similarity"] for entry in accuracy_log]
        return {
            "total_predictions": len(accuracy_log),
            "average_accuracy": np.mean(similarities),
            "min_accuracy": min(similarities),
            "max_accuracy": max(similarities),
            "last_evaluated": max(entry["evaluation_date"] for entry in accuracy_log)
        }


predictor = GeminiSportsPredictor()
evaluator = AccuracyEvaluator()

class PredictionRequest(BaseModel):
    keyword: str
    prediction_type: str
    sport: Optional[str] = None

@app.get("/api/sports")
async def get_supported_sports():
    return {"sports": predictor.valid_sports}

@app.post("/api/predict")
async def make_prediction(request: PredictionRequest):
    try:
        prediction = predictor.predict(
            request.prediction_type,
            request.keyword,
            request.sport
        )
        evaluator.log_prediction(prediction)
        return prediction
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)