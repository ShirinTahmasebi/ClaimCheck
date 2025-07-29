import dataclasses
from typing import Callable

from helpers.enums import LLMTypes, ExampleSelectionPolicy
from helpers.constants import RESULT_PATH_ZERO, RESULT_PATH_FEW
from helpers.prompts import SYSTEM_PROMPT_FLAT, INSTRUCTION_FLAT, INSTRUCTION_HIER_0, CLAIM_CATEGORIES, SUB_CLAIM_CATEGORIES
from output_processing import *

@dataclasses.dataclass
class TaskConfig:
    id: str
    name: str
    llm_type: LLMTypes
    is_flat: bool
    is_few_shot: bool
    instruction: str
    input_format: str
    process_output_fn: Callable[[str], str]
    result_dir: str
    val_mode: bool
    example_selection_policy: ExampleSelectionPolicy


ZeroConfig1 = TaskConfig(
    "zero_flat",
    "ClaimCheck::ZeroFlat",
    llm_type=LLMTypes.GEMMA,
    is_flat=True,
    is_few_shot=False,
    instruction=f"{SYSTEM_PROMPT_FLAT} \n {SUB_CLAIM_CATEGORIES} \n\n {INSTRUCTION_FLAT}",
    input_format="Input: {} \n\n Answer:",
    process_output_fn=process_output_flat,
    result_dir=RESULT_PATH_ZERO,
    val_mode=False,
    example_selection_policy=None
)

ZeroConfig2 = TaskConfig(
    "zero_hierarchy_0",
    "ClaimCheck::ZeroHier",
    llm_type=LLMTypes.GEMMA,
    is_flat=False,
    is_few_shot=False,
    instruction=f"{SYSTEM_PROMPT_FLAT} \n\n {CLAIM_CATEGORIES} \n\n {INSTRUCTION_HIER_0}",
    input_format="Input: {} \n\n Answer:",
    process_output_fn=process_output_zero_hier,
    result_dir=RESULT_PATH_ZERO,
    val_mode=True,
    example_selection_policy=None
)

FewConfig1 = TaskConfig(
    "few_similar_flat",
    "ClaimCheck::FewShotFlat::Similar",
    llm_type=LLMTypes.GEMMA,
    is_flat=True,
    is_few_shot=True,
    instruction=f"{SYSTEM_PROMPT_FLAT} \n\n {SUB_CLAIM_CATEGORIES} \n\n {INSTRUCTION_FLAT}",
    input_format="Input: {} \n\n Answer:",
    process_output_fn=process_output_flat,
    result_dir=RESULT_PATH_FEW,
    val_mode=True,
    example_selection_policy=ExampleSelectionPolicy.MOST_SIMILAR
)