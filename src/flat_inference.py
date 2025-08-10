from datasets import load_dataset
import os 
import pandas as pd

from helpers.constants import *
from task_config import *


if __name__ == "__main__":
    config_list = [
        ZeroFlatConfig1, ZeroFlatConfig2, 
        ZeroCoTFlatConfig1, ZeroCoTFlatConfig2,
        FewFlatConfig1, FewFlatConfig2, 
        FewStaticFlatConfig1, FewStaticFlatConfig2, 
        ZeroHierConfig1, ZeroHierConfig2
    ]

    parser = argparse.ArgumentParser(description="Subset processor")
    parser.add_argument("--start", type=int, required=False, help="Start index", default=0)
    parser.add_argument("--num", type=int, required=False, help="Number of items to process", default=100)
    parser.add_argument("--solution", type=str, required=False, help="The solution name")
    args = parser.parse_args()

    start_index = args.start
    end_index = start_index + args.num
    solution_name = args.solution
    task_config = [c for c in config_list if c.name == solution_name][0]

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
    result_path = f"{result_dir}/{task_config.id}_{start_index}_{end_index}.csv"
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
    

    end_index = min(end_index, len(data))
    for idx, item in enumerate(data.select(range(start_index, end_index))):
        id = idx + start_index
        print(f"Just started processing row {id}")
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
                "id": id,
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
                "id": id,
                "sub_claim_code": item["sub_claim_code"],
                "prediction": INVALID_PREDICTION,
                "text": item["text"],
                "output_text": ""
            })
        
        print(f"Just finished processing row {id}")

        if idx % 10 == 0:
            df = pd.DataFrame(results)
            df.to_csv(result_path, index=False)
        


    df = pd.DataFrame(results)
    df.to_csv(result_path, index=False)
