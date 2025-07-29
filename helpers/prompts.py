import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


SYSTEM_PROMPT_FLAT = """
You are an expert in climate misinformation detection. Given a climate-related claim, your task is to classify it into one of the predefined sub-claim categories listed below. 
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


from helpers.constants import LABEL_CLAIM_CATEGORIES, LABEL_SUB_CLAIM_CATEGORIES
claim_categories_text = "\n".join(f"{key}: {value}" for key, value in LABEL_CLAIM_CATEGORIES.items())
sub_claim_categories_text = "\n".join(f"{key}: {value}" for key, value in LABEL_SUB_CLAIM_CATEGORIES.items())


CLAIM_CATEGORIES = f"""
Here is the subclaim categories:

{claim_categories_text}
"""

SUB_CLAIM_CATEGORIES = f"""
{sub_claim_categories_text}
"""