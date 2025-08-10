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


# Static Examples
from collections import defaultdict

shortest_examples = defaultdict(lambda: {"text": None, "length": float("inf")})
for example in train_data:
    label_code = example["sub_claim_code"]
    label_text = example["sub_claim"]
    text = example["text"]
    text_length = len(text)
    
    if text in [
        # 0_0
        "Ken Smith (1957-2001): A Remembrance", 
        "By Anthony Watts, WUWT, Aug 11, 2015", 
        "Lisa Randall's Warped Passages website",
        "By Michael Bastasch, Daily Caller, Aug 11, 2015",
        "Rutgers University Climate Lab :: Global Snow Lab",
        "By Staff Writers, Press Association, Aug 13, 2015",
        "Study: No Increase in Northwest Australian Tropical Cyclones",
        # 2_1
        "The Sun's Impact On Earth's Temperature Goes Far Beyond TSI New Paper Shows",
        # 2_3
        "Answer: Nearly all the claims of evidence amount to ???effects?? of global warming, and not the cause. (See the missing hot spot.)"
    ]:
        continue
    
    if label_code == "0_0" and text_length < 100:
        continue

    if text_length < shortest_examples[label_code]["length"]:
        shortest_examples[label_code] = {"text": text, "length": text_length, "sub_claim_code": label_code, "sub_claim": label_text}

selected_examples = [v for k, v in sorted(shortest_examples.items())]

for ex in selected_examples:
    print(f"sub_claim_code: {ex['sub_claim_code']}\nsub_claim: {ex['sub_claim']}\nText: {ex['text']}\nLength: {ex['length']}\n{'-'*80}")

#### RAG Based

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