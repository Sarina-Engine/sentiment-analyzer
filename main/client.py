import grpc
from grpc_configs.prediction_pb2_grpc import PredictSentimentStub
from grpc_configs.prediction_pb2 import Comment

channel = grpc.insecure_channel("localhost:50051")
client = PredictSentimentStub(channel)

request = Comment(comment='اصلا خوب نبود اصلا پیشنهاد نمیکنم')

if __name__ == "__main__":
    response = client.Predict(request)
    print(response)
    print(response.digisentiment['digi'])
    print(response.digisentiment['digi'].sentiment[0])
