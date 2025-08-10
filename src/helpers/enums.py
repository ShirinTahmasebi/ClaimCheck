from enum import Enum

class LLMTypes(Enum):
    QWEN = "Qwen/Qwen2.5-3B-instruct"
    GEMMA = "google/gemma-2-2b-it"

class ExampleSelectionPolicy(Enum):
    RANDOM = "random"
    ONE_PER_CLASS = "one_per_class"
    MOST_SIMILAR = "most_similar"
    LEAST_SIMILAR = "least_similar"
    HYBRID = "hybrid"