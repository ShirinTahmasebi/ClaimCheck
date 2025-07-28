DATASET_BASE_URL = "https://huggingface.co/datasets/murathankurfali/ClimateEval/resolve/main/exeter/sub_claim"
DATASET_TRAIN_URL = f"{DATASET_BASE_URL}/training.csv"
DATASET_TEST_URL = f"{DATASET_BASE_URL}/test.csv"
DATASET_VALIDATION_URL = f"{DATASET_BASE_URL}/validation.csv"

from helpers.utils import get_absolute_path
RAG_PATH = get_absolute_path("data/rag_train_data")


LABEL_CLAIM_CATEGORIES = {
    "0": "No claim",
    "1": "Global warming is not happening",
    "2": "Human greenhouse gases are not causing climate change",
    "3": "Climate impacts/global warming is beneficial/not bad",
    "4": "Climate solutions won't work",
    "5": "Climate movement/science is unreliable"
}

LABEL_SUB_CLAIM_CATEGORIES = {
    "0_0": "No claim",
    "1_1": "Ice/permafrost/snow cover isn't melting",
    "1_2": "We're heading into an ice age/global cooling",
    "1_3": "Weather is cold/snowing",
    "1_4": "Climate hasn't warmed/changed over the last (few) decade(s)",
    "1_6": "Sea level rise is exaggerated/not accelerating",
    "1_7": "Extreme weather isn't increasing/has happened before/isn't linked to climate change",
    "2_1": "It's natural cycles/variation",
    "2_3": "There's no evidence for greenhouse effect/carbon dioxide driving climate change",
    "3_1": "Climate sensitivity is low/negative feedbacks reduce warming",
    "3_2": "Species/plants/reefs aren't showing climate impacts/are benefiting from climate change",
    "3_3": "CO2 is beneficial/not a pollutant",
    "4_1": "Climate policies (mitigation or adaptation) are harmful",
    "4_2": "Climate policies are ineffective/flawed",
    "4_4": "Clean energy technology/biofuels won't work",
    "4_5": "People need energy (e.g. from fossil fuels/nuclear)",
    "5_1": "Climate-related science is unreliable/uncertain/unsound (data, methods & models)",
    "5_2": "Climate movement is unreliable/alarmist/corrupt"
}