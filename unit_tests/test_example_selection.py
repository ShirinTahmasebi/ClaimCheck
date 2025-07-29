import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from few_shots_selector.sentence_embedder import MiniLMSentenceEmbedder
model = MiniLMSentenceEmbedder()

from datasets import load_dataset
from helpers.constants import *


data_files = {
    "train": DATASET_TRAIN_URL,
    "validation": DATASET_VALIDATION_URL,
    "test": DATASET_TEST_URL
}

dataset = load_dataset("csv", data_files=data_files)

train_data = dataset["train"].select(range(100))
val_data = dataset["validation"]
test_data = dataset["test"]



from few_shots_selector.rag_manager import RagManager
from helpers.constants import RAG_PATH
from helpers.enums import ExampleSelectionPolicy

rag = RagManager(
    train_data,
    RAG_PATH,
    model
)


input_sentence = "After a short protest from Massachusetts Republicans in their state Senate, the commonwealth is on the verge of changing its law to allow Gov. Deval Patrick (D) to appoint an interim Senator until the election to fill the late Sen. Edward Kennedy's spot can be held in January."
examples, labels_code, labels_text, scores = rag.select(input_sentence, ExampleSelectionPolicy.MOST_SIMILAR)

print(examples)
print(labels_code)
print(labels_text)
print(scores)