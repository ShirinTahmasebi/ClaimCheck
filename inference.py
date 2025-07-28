from datasets import load_dataset
from helpers.constants import *
data_files = {
    "train": DATASET_TRAIN_URL,
    "validation": DATASET_VALIDATION_URL,
    "test": DATASET_TEST_URL
}

dataset = load_dataset("csv", data_files=data_files)

train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]




from helpers.enums import LLMTypes
from llm_utils.llm_wrapper import LLMWrapper
llm_model = LLMWrapper.initialize_llm_by_type(LLMTypes.GEMMA)

from config import ZeroShotHier0Config

task_config = ZeroShotHier0Config

for item in val_data.select(range(10)):
    input_text = task_config.input_format.format(item["text"])
    instruction = task_config.instruction
    examples = []

    if not task_config.is_few_shot:
        examples = None

    text_output_1 = llm_model(instruction, examples, input_text)
    print(text_output_1)