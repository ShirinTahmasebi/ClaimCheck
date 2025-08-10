import dataclasses
from typing import Callable

from helpers.enums import LLMTypes, ExampleSelectionPolicy
from helpers.constants import RESULT_PATH_ZERO, RESULT_PATH_FEW, RESULT_PATH_FEW_STATIC
from helpers.prompts import SYSTEM_PROMPT_FLAT, SYSTEM_PROMPT_FLAT_COT, INSTRUCTION_FLAT, INSTRUCTION_FLAT_COT, INSTRUCTION_HIER_0, CLAIM_CATEGORIES, SUB_CLAIM_CATEGORIES, sub_claim_categories_by_claim_text
from output_processing import *

@dataclasses.dataclass
class TaskConfig:
    id: str
    name: str
    llm_type: LLMTypes
    is_few_shot: bool
    input_format: str
    result_dir: str
    val_mode: bool
    example_selection_policy: ExampleSelectionPolicy


@dataclasses.dataclass
class FlatTaskConfig(TaskConfig):
    instruction: str
    process_output_fn: Callable[[str], str]


@dataclasses.dataclass
class HierTaskConfig(TaskConfig):
    instruction_step_1: str
    instruction_step_2: str
    process_output_step_1_fn: Callable[[str], str]
    process_output_step_2_fn: Callable[[str], str]

ZeroFlatConfig1 = FlatTaskConfig(
    id="zero_flat",
    name="ClaimCheck::ZeroFlat",
    llm_type=LLMTypes.GEMMA,
    is_few_shot=False,
    instruction=f"{SYSTEM_PROMPT_FLAT} \n {SUB_CLAIM_CATEGORIES} \n\n {INSTRUCTION_FLAT}",
    input_format="Input: {} \n\n Answer:",
    process_output_fn=process_output_subclaim,
    result_dir=RESULT_PATH_ZERO,
    val_mode=False,
    example_selection_policy=None
)

ZeroFlatConfig2 = FlatTaskConfig(
    id="zero_flat",
    name="ClaimCheck::ZeroFlat::Val",
    llm_type=LLMTypes.GEMMA,
    is_few_shot=False,
    instruction=f"{SYSTEM_PROMPT_FLAT} \n {SUB_CLAIM_CATEGORIES} \n\n {INSTRUCTION_FLAT}",
    input_format="Input: {} \n\n Answer:",
    process_output_fn=process_output_subclaim,
    result_dir=RESULT_PATH_ZERO,
    val_mode=True,
    example_selection_policy=None
)

ZeroCoTFlatConfig1 = FlatTaskConfig(
    id="zero_flat",
    name="ClaimCheck::ZeroCoTFlat",
    llm_type=LLMTypes.GEMMA,
    is_few_shot=False,
    instruction=f"{SYSTEM_PROMPT_FLAT_COT} \n {SUB_CLAIM_CATEGORIES} \n\n {INSTRUCTION_FLAT_COT}",
    input_format="Input: {} \n\n Answer:",
    process_output_fn=process_output_subclaim,
    result_dir=RESULT_PATH_ZERO,
    val_mode=False,
    example_selection_policy=None
)

ZeroCoTFlatConfig2 = FlatTaskConfig(
    id="zero_flat",
    name="ClaimCheck::ZeroCoTFlat::Val",
    llm_type=LLMTypes.GEMMA,
    is_few_shot=False,
    instruction=f"{SYSTEM_PROMPT_FLAT_COT} \n {SUB_CLAIM_CATEGORIES} \n\n {INSTRUCTION_FLAT_COT}",
    input_format="Input: {} \n\n Answer:",
    process_output_fn=process_output_subclaim,
    result_dir=RESULT_PATH_ZERO,
    val_mode=True,
    example_selection_policy=None
)

FewFlatConfig1 = FlatTaskConfig(
    id="few_similar_flat",
    name="ClaimCheck::FewShotFlat::Similar",
    llm_type=LLMTypes.GEMMA,
    is_few_shot=True,
    instruction=f"{SYSTEM_PROMPT_FLAT} \n\n {SUB_CLAIM_CATEGORIES} \n\n {INSTRUCTION_FLAT}",
    input_format="Input: {} \n\n Answer:",
    process_output_fn=process_output_subclaim,
    result_dir=RESULT_PATH_FEW,
    val_mode=False,
    example_selection_policy=ExampleSelectionPolicy.MOST_SIMILAR
)

FewFlatConfig2 = FlatTaskConfig(
    id="few_similar_flat",
    name="ClaimCheck::FewShotFlat::Similar::Val",
    llm_type=LLMTypes.GEMMA,
    is_few_shot=True,
    instruction=f"{SYSTEM_PROMPT_FLAT} \n\n {SUB_CLAIM_CATEGORIES} \n\n {INSTRUCTION_FLAT}",
    input_format="Input: {} \n\n Answer:",
    process_output_fn=process_output_subclaim,
    result_dir=RESULT_PATH_FEW,
    val_mode=True,
    example_selection_policy=ExampleSelectionPolicy.MOST_SIMILAR
)

FewStaticFlatConfig1 = FlatTaskConfig(
    id="few_static_flat",
    name="ClaimCheck::FewShotFlat::Static",
    llm_type=LLMTypes.GEMMA,
    is_few_shot=True,
    instruction=f"{SYSTEM_PROMPT_FLAT} \n\n {SUB_CLAIM_CATEGORIES} \n\n {INSTRUCTION_FLAT}",
    input_format="Input: {} \n\n Answer:",
    process_output_fn=process_output_subclaim,
    result_dir=RESULT_PATH_FEW_STATIC,
    val_mode=False,
    example_selection_policy=ExampleSelectionPolicy.ONE_PER_CLASS
)

FewStaticFlatConfig2 = FlatTaskConfig(
    id="few_static_flat",
    name="ClaimCheck::FewShotFlat::Static::Val",
    llm_type=LLMTypes.GEMMA,
    is_few_shot=True,
    instruction=f"{SYSTEM_PROMPT_FLAT} \n\n {SUB_CLAIM_CATEGORIES} \n\n {INSTRUCTION_FLAT}",
    input_format="Input: {} \n\n Answer:",
    process_output_fn=process_output_subclaim,
    result_dir=RESULT_PATH_FEW_STATIC,
    val_mode=True,
    example_selection_policy=ExampleSelectionPolicy.ONE_PER_CLASS
)

ZeroHierConfig1 = HierTaskConfig(
    id="zero_hierarchy",
    name="ClaimCheck::ZeroHier",
    llm_type=LLMTypes.GEMMA,
    is_few_shot=False,
    instruction_step_1=f"{SYSTEM_PROMPT_FLAT} \n\n {CLAIM_CATEGORIES} \n\n {INSTRUCTION_HIER_0}",
    instruction_step_2=lambda x: f"{SYSTEM_PROMPT_FLAT} \n\n {sub_claim_categories_by_claim_text(x)} \n\n {INSTRUCTION_HIER_0}",
    input_format="Input: {} \n\n Answer:",
    process_output_step_1_fn=process_output_claim,
    process_output_step_2_fn=process_output_subclaim,
    result_dir=RESULT_PATH_ZERO,
    val_mode=False,
    example_selection_policy=None
)

ZeroHierConfig2 = HierTaskConfig(
    id="zero_hierarchy",
    name="ClaimCheck::ZeroHier::Val",
    llm_type=LLMTypes.GEMMA,
    is_few_shot=False,
    instruction_step_1=f"{SYSTEM_PROMPT_FLAT} \n\n {CLAIM_CATEGORIES} \n\n {INSTRUCTION_HIER_0}",
    instruction_step_2=lambda x: f"{SYSTEM_PROMPT_FLAT} \n\n {sub_claim_categories_by_claim_text(x)} \n\n {INSTRUCTION_HIER_0}",
    input_format="Input: {} \n\n Answer:",
    process_output_step_1_fn=process_output_claim,
    process_output_step_2_fn=process_output_subclaim,
    result_dir=RESULT_PATH_ZERO,
    val_mode=True,
    example_selection_policy=None
)
