from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Binarysentiment(_message.Message):
    __slots__ = ["sentiment"]
    SENTIMENT_FIELD_NUMBER: _ClassVar[int]
    sentiment: _containers.RepeatedCompositeFieldContainer[Sentiment]
    def __init__(self, sentiment: _Optional[_Iterable[_Union[Sentiment, _Mapping]]] = ...) -> None: ...

class Comment(_message.Message):
    __slots__ = ["comment"]
    COMMENT_FIELD_NUMBER: _ClassVar[int]
    comment: str
    def __init__(self, comment: _Optional[str] = ...) -> None: ...

class Digisentiment(_message.Message):
    __slots__ = ["sentiment"]
    SENTIMENT_FIELD_NUMBER: _ClassVar[int]
    sentiment: _containers.RepeatedCompositeFieldContainer[Sentiment]
    def __init__(self, sentiment: _Optional[_Iterable[_Union[Sentiment, _Mapping]]] = ...) -> None: ...

class Multisentiment(_message.Message):
    __slots__ = ["sentiment"]
    SENTIMENT_FIELD_NUMBER: _ClassVar[int]
    sentiment: _containers.RepeatedCompositeFieldContainer[Sentiment]
    def __init__(self, sentiment: _Optional[_Iterable[_Union[Sentiment, _Mapping]]] = ...) -> None: ...

class Prediction(_message.Message):
    __slots__ = ["binarysentiment", "comment", "digisentiment", "mulitsentiment", "snappsentiment"]
    class BinarysentimentEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Binarysentiment
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Binarysentiment, _Mapping]] = ...) -> None: ...
    class DigisentimentEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Digisentiment
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Digisentiment, _Mapping]] = ...) -> None: ...
    class MulitsentimentEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Multisentiment
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Multisentiment, _Mapping]] = ...) -> None: ...
    class SnappsentimentEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Snappsentiment
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Snappsentiment, _Mapping]] = ...) -> None: ...
    BINARYSENTIMENT_FIELD_NUMBER: _ClassVar[int]
    COMMENT_FIELD_NUMBER: _ClassVar[int]
    DIGISENTIMENT_FIELD_NUMBER: _ClassVar[int]
    MULITSENTIMENT_FIELD_NUMBER: _ClassVar[int]
    SNAPPSENTIMENT_FIELD_NUMBER: _ClassVar[int]
    binarysentiment: _containers.MessageMap[str, Binarysentiment]
    comment: str
    digisentiment: _containers.MessageMap[str, Digisentiment]
    mulitsentiment: _containers.MessageMap[str, Multisentiment]
    snappsentiment: _containers.MessageMap[str, Snappsentiment]
    def __init__(self, comment: _Optional[str] = ..., digisentiment: _Optional[_Mapping[str, Digisentiment]] = ..., snappsentiment: _Optional[_Mapping[str, Snappsentiment]] = ..., binarysentiment: _Optional[_Mapping[str, Binarysentiment]] = ..., mulitsentiment: _Optional[_Mapping[str, Multisentiment]] = ...) -> None: ...

class Sentiment(_message.Message):
    __slots__ = ["label", "score"]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    label: str
    score: float
    def __init__(self, label: _Optional[str] = ..., score: _Optional[float] = ...) -> None: ...

class Snappsentiment(_message.Message):
    __slots__ = ["sentiment"]
    SENTIMENT_FIELD_NUMBER: _ClassVar[int]
    sentiment: _containers.RepeatedCompositeFieldContainer[Sentiment]
    def __init__(self, sentiment: _Optional[_Iterable[_Union[Sentiment, _Mapping]]] = ...) -> None: ...
