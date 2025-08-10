def process_output_claim(llm_response: str) -> str:
    import re
    from helpers.constants import INVALID_PREDICTION

    found_numbers = list(set(map(int, re.findall(r'\b[0-5]+\b', llm_response))))
    if len(found_numbers) != 1:
        print(f"A suspicous response is found. The found numbers are: " + str(found_numbers))
        found_numbers = [INVALID_PREDICTION]
    return found_numbers[0] 


def process_output_subclaim(llm_response: str) -> str:
    import re
    from helpers.constants import LABEL_SUB_CLAIM_CATEGORIES, INVALID_PREDICTION

    matches = re.findall(r'\b\d+_\d+\b', llm_response)
    allowed_matches = list(set([match for match in matches if match in LABEL_SUB_CLAIM_CATEGORIES]))

    if len(allowed_matches) != 1:
        print(f"A suspicous response is found. The found numbers are: " + str(allowed_matches))
        allowed_matches = [INVALID_PREDICTION]
    
    return allowed_matches[0]