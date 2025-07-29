from datasets import load_dataset
import os 
import pandas as pd

from helpers.constants import *
from task_config import ZeroConfig1

task_config = ZeroConfig1


data_files = {
    "train": DATASET_TRAIN_URL,
    "validation": DATASET_VALIDATION_URL,
    "test": DATASET_TEST_URL
}

dataset = load_dataset("csv", data_files=data_files)
data = dataset["validation" if task_config.val_mode else "test"]
train_data = dataset["train"]


from llm_utils.llm_wrapper import LLMWrapper
llm_model = LLMWrapper.initialize_llm_by_type(task_config.llm_type)

result_dir = task_config.result_dir.format("val" if task_config.val_mode else "test") 
result_path = f"{result_dir}/{task_config.id}.csv"
os.makedirs(result_dir, exist_ok=True)


if task_config.is_few_shot:
    from few_shots_selector.sentence_embedder import MiniLMSentenceEmbedder
    from few_shots_selector.rag_manager import RagManager
    from helpers.constants import RAG_PATH
    model = MiniLMSentenceEmbedder()
    rag = RagManager(train_data, RAG_PATH, model)
else:
    rag = None

results = []
for idx, item in enumerate(data):
    print(f"Just started processing row {idx}")
    try:
        input_text = task_config.input_format.format(item["text"])
        instruction = task_config.instruction
        example_list = []

        if not task_config.is_few_shot:
            example_list = None
        else:
            examples, labels_code, labels_text, _ = rag.select(input_text, task_config.example_selection_policy)
            example_list = []
            for i in range(len(examples)):
                example_list.append({"text": examples[i], "label": f"{labels_code[i]}: {labels_text[i]}"})

        text_output = llm_model(instruction, example_list, input_text)
        selected_category = task_config.process_output_fn(text_output)

        results.append({
            "id": idx,
            "sub_claim_code": item["sub_claim_code"],
            "prediction": selected_category,
            "text": item["text"],
            "output_text": text_output
        })
    except Exception as ex:
        import sys
        import os
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(ex)
        results.append({
            "id": idx,
            "sub_claim_code": item["sub_claim_code"],
            "prediction": INVALID_PREDICTION,
            "text": item["text"],
            "output_text": ""
        })
    
    print(f"Just finished processing row {idx}")

    if idx % 10 == 0:
        df = pd.DataFrame(results)
        df.to_csv(result_path, index=False)
    


df = pd.DataFrame(results)
df.to_csv(result_path, index=False)
