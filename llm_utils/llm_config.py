import dataclasses

@dataclasses.dataclass
class LLMConfig:
    max_new_token: int
    temperature: float
    top_p: float
    top_k: float


QwenConfig = LLMConfig(
    max_new_token=1000,
    top_k=50,
    top_p=1,
    temperature=1
)

GemmaConfig = LLMConfig(
    max_new_token=1000,
    top_k=50,
    top_p=1,
    temperature=1
)
