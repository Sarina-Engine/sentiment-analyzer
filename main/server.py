import grpc
from concurrent import futures
from grpc_configs.prediction_pb2_grpc import PredictSentimentServicer
from grpc_configs.prediction_pb2 import (Prediction,
                            Digisentiment,
                            Snappsentiment,
                            Binarysentiment,
                            Multisentiment,
                            Sentiment
                            )
from grpc_configs import prediction_pb2_grpc

from torch.nn.functional import softmax
from src.data import Preprocessor
from src.utils import get_data_from_loader, id2label
from src import configs

class PredictSentiments(PredictSentimentServicer):
    def __init__(self):
        super().__init__()
        self.preprocess = Preprocessor()
        self.sentiment_models = configs.sentiment_models
        self.tokenizers = configs.tokenizers


    def Predict(self, request, context):
            text = request.comment

            text = self.preprocess(text)
            scores_dict = {}
            for model in configs.models_list:
                comment, comment_id, input_ids, attention_mask, token_type_ids = get_data_from_loader(text, self.tokenizers[model])
                outputs = self.sentiment_models[model](input_ids, attention_mask, token_type_ids)
                scores = softmax(outputs[0], dim=1).squeeze().detach()
                scores_dict[model] = scores

                           
            digi_sentiment = Digisentiment(sentiment=[
                    Sentiment(label='recommended', score=scores_dict['DIGIKALA'][2].item()),
                    Sentiment(label='not_recommended', score=scores_dict['DIGIKALA'][1].item()),
                    Sentiment(label='no_idea', score=scores_dict['DIGIKALA'][0].item())
                ]),
            snapp_sentiment = Snappsentiment(sentiment=[
                    Sentiment(label='sad', score=scores_dict['SNAPPFOOD'][1].item()),
                    Sentiment(label='happy', score=scores_dict['SNAPPFOOD'][0].item()),
                ]),
            binary_sentiment = Binarysentiment(sentiment=[
                    Sentiment(label='positive', score=scores_dict['DEEPSENTIPERS_BINARY'][1].item()),
                    Sentiment(label='negative', score=scores_dict['DEEPSENTIPERS_BINARY'][0].item()),
                ]),
            multi_sentiment = Multisentiment(sentiment=[
                    Sentiment(label='furious', score=scores_dict['DEEPSENTIPERS_MULTI'][0].item()),
                    Sentiment(label='angry', score=scores_dict['DEEPSENTIPERS_MULTI'][1].item()),
                    Sentiment(label='neutral', score=scores_dict['DEEPSENTIPERS_MULTI'][2].item()),
                    Sentiment(label='happy', score=scores_dict['DEEPSENTIPERS_MULTI'][3].item()),
                    Sentiment(label='delighted', score=scores_dict['DEEPSENTIPERS_MULTI'][4].item()),
                ])

            return Prediction(comment=text, 
                              digisentiment={'digi':digi_sentiment}, 
                              snappsentiment={'snapp':snapp_sentiment},
                              binarysentiment={'binary':binary_sentiment},
                              mulitsentiment={'multi':multi_sentiment})


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    prediction_pb2_grpc.add_PredictSentimentServicer_to_server(PredictSentiments(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print('Server is running...')
    server.wait_for_termination()


if __name__ == "__main__":
    serve()