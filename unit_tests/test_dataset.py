from datasets import load_dataset
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


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

print(train_data[0]['text'])
print(f"Size of train: {len(train_data)}")
print(f"Size of val: {len(val_data)}")
print(f"Size of test: {len(test_data)}")

print(train_data.info.description)
print(train_data.info.features) 

print(list(train_data.select([0,1,2])["text"]))

for item in val_data.select(range(10)):
    print(item["text"])
    # print(item["sub_claim"])
    print(item["sub_claim_code"])
