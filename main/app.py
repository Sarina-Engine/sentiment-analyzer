from io import StringIO
from flask import Flask, request, jsonify
from torch.nn.functional import softmax
from .src.data import Preprocessor
from .src.utils import get_data_from_loader, id2label
from .src import configs


app = Flask(__name__)

preprocess = Preprocessor()

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    if request.method == 'POST':
        text = request.args['text']

        try:
            text = preprocess(text)

            scores_dict = {}
            for model in configs.models_list:
                comment, comment_id, input_ids, attention_mask, token_type_ids = get_data_from_loader(text, configs.tokenizers[model])
                outputs = configs.sentiment_models[model](input_ids, attention_mask, token_type_ids)
                scores = softmax(outputs[0], dim=1).squeeze().detach()
                scores_dict[model] = scores

            # _, preds = torch.max(outputs[0], dim=1)
            # pred = preds.item()
            # label = id2label(pred)

            return jsonify({'res':{
                'comment': text,
                'sentimens': {                
                    'digi_sentiment': [
                        {
                        'label': 'recommended',
                        'score': scores_dict['DIGIKALA'][2].item()
                        },
                        {
                        'label':'not_recommended',
                        'score': scores_dict['DIGIKALA'][1].item()
                        },
                        {
                        'label':'no_idea',
                        'score': scores_dict['DIGIKALA'][0].item()
                        }
                    ],
                    'snapp_sentiment':[
                        {
                        'label': 'sad',
                        'score': scores_dict['SNAPPFOOD'][1].item()
                        },
                        {
                        'label':'happy',
                        'score': scores_dict['SNAPPFOOD'][0].item()
                        },
                    ],
                    'binary_sentiment':[
                        {
                        'label': 'positive',
                        'score': scores_dict['DEEPSENTIPERS_BINARY'][1].item()
                        },
                        {
                        'label':'negative',
                        'score': scores_dict['DEEPSENTIPERS_BINARY'][0].item()
                        },
                    ],
                    'multi_sentiment': [
                        {
                        'label': 'furious',
                        'score': scores_dict['DEEPSENTIPERS_MULTI'][0].item()
                        },
                        {
                        'label':'angry',
                        'score': scores_dict['DEEPSENTIPERS_MULTI'][1].item()
                        },
                                                {
                        'label': 'neutral',
                        'score': scores_dict['DEEPSENTIPERS_MULTI'][2].item()
                        },
                        {
                        'label':'happy',
                        'score': scores_dict['DEEPSENTIPERS_MULTI'][3].item()
                        },
                                                {
                        'label': 'delighted',
                        'score': scores_dict['DEEPSENTIPERS_MULTI'][4].item()
                        },
                    ]}
            }})

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