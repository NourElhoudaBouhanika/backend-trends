import os
import torch
import pandas as pd
import numpy as np
from torch import nn
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
from datetime import datetime


class TrendPredictor:
    def __init__(self, model_dir='./model_save'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
        self.bert_model = AutoModel.from_pretrained("asafaya/bert-base-arabic").to(self.device)

        # Load model components
        self.model, self.trend_encoder = self._load_model()

    def _load_model(self):
        """Load the saved model components"""
        # Load trend encoder
        encoder_info = torch.load(f'{self.model_dir}/trend_encoder_info.pth',
                                  map_location=self.device,
                                  weights_only=False)
        trend_classes = encoder_info['trend_classes']

        encoder = LabelEncoder()
        encoder.classes_ = trend_classes

        # Initialize and load model
        model = TrendPredictionModel(self.bert_model, len(trend_classes))
        model.load_state_dict(torch.load(f'{self.model_dir}/trend_prediction_model.pth',
                                         map_location=self.device))
        model.to(self.device)
        model.eval()

        return model, encoder

    def predict_trends(self, keyword: str, date: str = None, top_n: int = 5):
        """
        Predict trends for a given keyword and date
        Args:
            keyword: Input keyword to predict trends for
            date: Date string in format 'YYYY-MM-DD' (defaults to today)
            top_n: Number of top trends to return
        Returns:
            List of tuples (trend, probability)
        """
        if date is None:
            target_date = datetime.now()
        else:
            try:
                target_date = pd.to_datetime(date)
            except:
                target_date = datetime.now()

        month = target_date.month
        month_normalized = (month - 1) / 11

        # Tokenize input
        encoding = self.tokenizer(
            keyword,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        month_tensor = torch.tensor([[month_normalized]], dtype=torch.float).to(self.device)

        # Predict trends
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, month_tensor)

        # Get top_n predicted trends
        probabilities = outputs.cpu().numpy()[0]
        top_indices = np.argsort(probabilities)[-top_n:][::-1]

        # Convert indices to trends
        top_trends = [(self.trend_encoder.classes_[idx], float(probabilities[idx]))
                      for idx in top_indices]

        return top_trends


# Model definition (same as your teammate's)
class TrendPredictionModel(nn.Module):
    def __init__(self, bert_model, num_trends):
        super(TrendPredictionModel, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768 + 1, num_trends)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, month):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        cls_output = outputs.last_hidden_state[:, 0, :]
        combined = torch.cat([cls_output, month], dim=1)
        x = self.dropout(combined)
        x = self.classifier(x)
        return self.sigmoid(x)