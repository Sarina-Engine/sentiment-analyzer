cd main/grpc_configs
python -m grpc_tools.protoc -I protobufs --python_out=. --grpc_python_out=. protobufs/prediction.proto