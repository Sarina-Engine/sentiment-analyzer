import torch
from torch import nn
from transformers import BertModel, AutoModelForSequenceClassification

class SentimentModel(nn.Module):
    def __init__(self, model_path, config):
        super(SentimentModel, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_path, return_dict=False)

    def forward(self, input_ids, attention_mask, token_type_ids):
        pooled_output = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids)

        return pooled_output  