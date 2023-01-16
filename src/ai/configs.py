import os
import pathlib
from transformers import BertConfig, AutoTokenizer
from .utils import load_model

ABS_PATH = pathlib.Path().resolve()

OUTPUT_PATH = os.path.join(ABS_PATH, "src/ai/model/pretrained_models")

models_list = [
    'DIGIKALA',
    'SNAPPFOOD',
    'DEEPSENTIPERS_BINARY',
    'DEEPSENTIPERS_MULTI'
]

api_paths = {
    models_list[0] : "HooshvareLab/bert-fa-base-uncased-sentiment-digikala",
    models_list[1] : "HooshvareLab/bert-fa-base-uncased-sentiment-snappfood",
    models_list[2] : "HooshvareLab/bert-fa-base-uncased-sentiment-deepsentipers-binary",
    models_list[3] : "HooshvareLab/bert-fa-base-uncased-sentiment-deepsentipers-multi"
}

local_paths = {
    models_list[0] : os.path.join(OUTPUT_PATH, 'digi_model.pth'),
    models_list[1] : os.path.join(OUTPUT_PATH, 'snapp_model.pth'),
    models_list[2] : os.path.join(OUTPUT_PATH, 'binary_model.pth'),
    models_list[3] : os.path.join(OUTPUT_PATH, 'mutli_model.pth')
}

tokenizers = {
    models_list[0] : AutoTokenizer.from_pretrained(api_paths[models_list[0]]),
    models_list[1] : AutoTokenizer.from_pretrained(api_paths[models_list[1]]),
    models_list[2] : AutoTokenizer.from_pretrained(api_paths[models_list[2]]),
    models_list[3] : AutoTokenizer.from_pretrained(api_paths[models_list[3]])
}

model_configs = {
    models_list[0] : BertConfig.from_pretrained(api_paths[models_list[0]]),
    models_list[1] : BertConfig.from_pretrained(api_paths[models_list[1]]),
    models_list[2] : BertConfig.from_pretrained(api_paths[models_list[2]]),
    models_list[3] : BertConfig.from_pretrained(api_paths[models_list[3]])
}

sentiment_models = {
    models_list[0] : load_model(api_paths[models_list[0]], local_paths[models_list[0]], model_configs[models_list[0]]),
    models_list[1] : load_model(api_paths[models_list[1]], local_paths[models_list[1]], model_configs[models_list[1]]),
    models_list[2] : load_model(api_paths[models_list[2]], local_paths[models_list[2]], model_configs[models_list[2]]),
    models_list[3] : load_model(api_paths[models_list[3]], local_paths[models_list[3]], model_configs[models_list[3]])
}

