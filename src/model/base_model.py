import torch
from torch import nn
from transformers import BertModel, AutoModelForSequenceClassification

MODEL_NAME_OR_PATH = "HooshvareLab/bert-fa-base-uncased-sentiment-digikala"

class SentimentModel(nn.Module):
    def __init__(self, config):
        super(SentimentModel, self).__init__()

        self.bert = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_OR_PATH, return_dict=False)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        pooled_output = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids)
        
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        # return logits

        return pooled_output 