import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


SYSTEM_PROMPT_FLAT = """
You are an expert in climate misinformation detection. Given a climate-related claim, your task is to classify it into one of the predefined sub-claim categories listed below. 
"""


SYSTEM_PROMPT_FLAT_COT = """
You are an expert in climate misinformation detection. Given a climate-related claim, your task is to classify it into one of the predefined sub-claim categories listed below. 
First, analyze whether the claim rejects, casts doubt on, or misrepresents the scientific consensus on human-caused climate change. 
Then, think step-by-step to determine the reasoning behind the claim and identify the most suitable sub-claim category.
"""


INSTRUCTION_FLAT = """
Contrarian claims are statements that reject, cast doubt on, or misrepresent the scientific consensus on human-caused climate change.

If the claim does not express a contrarian view, return 0_0: No Claim.

Choose only the most appropriate category and GENERATE ONLY THE NUMBER CORRESPONDING TO THAT CATEGORY. DO NOT GENERATE ANY EXPLANATION.
"""

INSTRUCTION_HIER_0 = """
Contrarian claims are statements that reject, cast doubt on, or misrepresent the scientific consensus on human-caused climate change.

If the claim does not express a contrarian view, return 0: No Claim.

Choose only the most appropriate category and GENERATE ONLY THE NUMBER CORRESPONDING TO THAT CATEGORY. DO NOT GENERATE ANY EXPLANATION.
"""


INSTRUCTION_FLAT_COT = """
Contrarian claims are statements that reject, cast doubt on, or misrepresent the scientific consensus on human-caused climate change.

If the claim does not express a contrarian view, return 0_0: No Claim.

Follow this format:
Step-by-step reasoning: <model generates rationale here>
Final answer: <ONLY the category number>

DO NOT generate any extra text or explanations beyond this format.
Think carefully before answering. Do not skip steps. Be concise but logical.
"""


from helpers.constants import LABEL_CLAIM_CATEGORIES, LABEL_SUB_CLAIM_CATEGORIES
claim_categories_text = "\n".join(f"{key}: {value}" for key, value in LABEL_CLAIM_CATEGORIES.items())
sub_claim_categories_text = "\n".join(f"{key}: {value}" for key, value in LABEL_SUB_CLAIM_CATEGORIES.items())
sub_claim_categories_by_claim_text = lambda x: "\n".join(f"{key}: {value}" for key, value in LABEL_SUB_CLAIM_CATEGORIES.items() if key.split("_")[0] == str(x))

CLAIM_CATEGORIES = f"""
Here is the subclaim categories:

{claim_categories_text}
"""

SUB_CLAIM_CATEGORIES = f"""
{sub_claim_categories_text}
"""