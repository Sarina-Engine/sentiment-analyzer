from io import StringIO
from flask import Flask, request, jsonify
import torch 
from torch.nn.functional import softmax
from data import Preprocessor
from utils import load_model, get_data_from_loader, id2label


app = Flask(__name__)

model = load_model()
preprocess = Preprocessor()

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    if request.method == 'POST':
        text = request.get_json()['text']

        try:
            text = preprocess(text)
            comment, comment_id, input_ids, attention_mask, token_type_ids = get_data_from_loader(text)
            outputs = model(input_ids, attention_mask, token_type_ids)

            scores = softmax(outputs[0], dim=1).squeeze().detach()
            _, preds = torch.max(outputs[0], dim=1)

            pred = preds.item()
            label = id2label(pred)

            return jsonify({
                'res': [
                    {
                    'label': 'recommended',
                    'score': scores[2].item()
                    },
                    {
                    'label':'no_idea',
                    'score': scores[1].item()
                    },
                    {
                    'label':'not recommended',
                    'score': scores[0].item()
                    }
                ]
            })

        except Exception as e:
            print(e)





        # commentt = data.iloc[comment_id]['comment'].to_json()

        # return jsonify({'res': {
        #         'pred_id': pred,
        #         'label': label,
        #         'comment': comment,
        #         'comment_id': comment_id.item(),
        #         'score': scores[0][pred].item()
        #         }}
        #     )