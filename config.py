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


@dataclasses.dataclass
class TaskConfig:
    name: str
    is_flat: bool
    is_few_shot: bool
    instruction: str
    input_format: str


from helpers.prompts import INSTRUCTION_ZEROSHOT_FLAT, INSTRUCTION_ZEROSHOT_HIER_0, CLAIM_CATEGORIES, SUB_CLAIM_CATEGORIES
ZeroShotFlatConfig = TaskConfig(
    "ClaimChecl_zero_flat",
    is_flat=True,
    is_few_shot=False,
    instruction=f"{INSTRUCTION_ZEROSHOT_FLAT} \n\n {SUB_CLAIM_CATEGORIES}",
    input_format="Input: {} \n\n Response:"
)

ZeroShotHier0Config = TaskConfig(
    "ClaimChecl_zero_hierarchy_0",
    is_flat=False,
    is_few_shot=False,
    instruction=f"{INSTRUCTION_ZEROSHOT_HIER_0} \n\n {CLAIM_CATEGORIES}",
    input_format="Input: {} \n\n Response:"
)