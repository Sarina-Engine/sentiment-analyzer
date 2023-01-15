import os
import torch
import numpy as np
from .model import SentimentModel
from .data import create_data_loader

MAX_LEN = 128
BATCH_SIZE = 1

def save_model(api_path, path, config):
    model = SentimentModel(model_path=api_path, config=config)
    torch.save(model.state_dict(), path)

def load_model(api_path, path, config):
    if os.path.exists(path):
        model = SentimentModel(model_path=api_path, config=config)
        model.load_state_dict(torch.load(path))
        model.eval()
    else:
        save_model(api_path, path, config)
        return load_model(api_path, path, config)

    return model


def get_data_from_loader(text, tokenizer):
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