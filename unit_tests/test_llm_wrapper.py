import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from helpers.enums import LLMTypes
from llm_utils.llm_wrapper import LLMWrapper

input_text = "Give me a short introduction to large language models."

hf_model_1 = LLMWrapper.initialize_llm_by_type(LLMTypes.QWEN)
text_output_1 = hf_model_1(None, None, input_text)
print(text_output_1)


hf_model_2 = LLMWrapper.initialize_llm_by_type(LLMTypes.QWEN, use_chat_style=True)
text_output_2 = hf_model_2(None, None, input_text)
print(text_output_2)


hf_model_3 = LLMWrapper.initialize_llm_by_type(LLMTypes.GEMMA)
text_output_3 = hf_model_3(None, None, input_text)
print(text_output_3)

