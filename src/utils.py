import os
import torch
import numpy as np
from transformers import BertConfig, AutoTokenizer
from model import SentimentModel
from data import create_data_loader

MODEL_NAME_OR_PATH = "HooshvareLab/bert-fa-base-uncased-sentiment-digikala"
OUTPUT_PATH = "model/pretrained_models"
path = os.path.join(OUTPUT_PATH, 'sentiment_model.pth')
MAX_LEN = 128
BATCH_SIZE = 1

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
config = BertConfig.from_pretrained(MODEL_NAME_OR_PATH)

def save_model():
    model = SentimentModel(config=config)
    torch.save(model.state_dict(), path)

def load_model():
    if os.path.exists(path):
        model = SentimentModel(config=config)
        model.load_state_dict(torch.load(path))
        model.eval()
    else:
        save_model()
        return load_model()

    return model

def get_data_from_loader(text):
    # data_loader = create_data_loader(data['cleaned_comment'].to_numpy(), tokenizer, MAX_LEN, BATCH_SIZE)
    data_loader = create_data_loader(np.array([text]), tokenizer, MAX_LEN, BATCH_SIZE)
    sample_data = next(iter(data_loader))

    sample_data_comment = sample_data['comment']
    sample_data_comment_id = sample_data['comment_id']
    sample_data_input_ids = sample_data['input_ids']
    sample_data_attention_mask = sample_data['attention_mask']
    sample_data_token_type_ids = sample_data['token_type_ids']

    return (
        sample_data_comment,
        sample_data_comment_id,
        sample_data_input_ids,
        sample_data_attention_mask,
        sample_data_token_type_ids
    )

def id2label(id):
    dictt = {
        1: 'no_idea',
        2: 'recommended',
        0: 'not_recommended'
    }

    return dictt[id]