from abc import ABC, abstractmethod

class SentenceEmbedderWrapper(ABC):
    def __init__(self):
        pass

    def __call__(self, sentences:list[str]):
        return self._encode_sentences(sentences)

    @abstractmethod
    def _encode_sentences(self, sentences:list[str]):
        raise NotImplementedError(f"The sentence encodder is not defined for {self.__class__.__name__}.")


class MiniLMSentenceEmbedder(SentenceEmbedderWrapper):
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def _encode_sentences(self, sentences:list[str]):
        return self.model.encode(sentences, convert_to_numpy=True)
