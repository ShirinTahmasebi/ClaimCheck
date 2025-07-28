from abc import ABC, abstractmethod
import torch

import sys
sys.path.append("..")

from config import LLMConfig, QwenConfig, GemmaConfig

class LLMWrapper(ABC):
    
    def __init__(self, model_name, access_token, use_chat_style):
        super().__init__()
        self.model_name = model_name
        self.access_token = access_token
        self.use_chat_style = use_chat_style
        self.model, self.tokenizer = self.initialize_model_tokenizer()

    def __call__(self, instruction, examples, input):
        return self.call_llm(instruction, examples, input)
        
    @classmethod
    def initialize_llm_by_type(cls, llm_type, use_chat_style=False):
        from helpers.access_keys import ACCESS_KEYS
        from helpers.enums import LLMTypes
        if llm_type == LLMTypes.QWEN:
            return QwenModel(LLMTypes.QWEN.value, ACCESS_KEYS.HF_TOKEN, use_chat_style)
        if llm_type == LLMTypes.GEMMA:
            return GemmaModel(LLMTypes.GEMMA.value, ACCESS_KEYS.HF_TOKEN, use_chat_style)
       
    def call_llm(self, instruction, examples, input) -> str:
        formatted_text = self.format_prompt(instruction, examples, input)
        print(f"The formatted text passed to the llm is: \n {formatted_text}")
        print("-------")
        raw_response = self.generate(formatted_text)
        final_response = self.post_process_response(raw_response)
        return final_response

    @abstractmethod
    def initialize_model_tokenizer(self):
        raise NotImplementedError(f"The model initialization method is not implemented for {self.__class__.__name__}.")
    
    @abstractmethod
    def get_generation_config(self) -> LLMConfig:
        raise NotImplementedError(f"The generation config is not defined for {self.__class__.__name__}.")
    
    @abstractmethod
    def format_prompt(self, instruction:str, examples:list[str], input:str) -> str:
        raise NotImplementedError(f"The prompt formatter is not defined for {self.__class__.__name__}.")
    
    @abstractmethod
    def generate(self, input_text) -> str:
        raise NotImplementedError(f"The model text generation method is not defined for {self.__class__.__name__}.")

    @abstractmethod
    def post_process_response(self, raw_response) -> str:
        raise NotImplementedError(f"The model text generation method is not defined for {self.__class__.__name__}.")



class HFModels(LLMWrapper, ABC):
    def initialize_model_tokenizer(self):
        from huggingface_hub import login
        login(self.access_token)
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        return model, tokenizer


    def generate(self, input_text: str) -> str:
        input_ids = self.tokenizer(input_text, return_tensors='pt').input_ids
        outputs = self.model.generate(input_ids=input_ids, generation_config=self.get_generation_config())
        output = outputs[0][len(input_ids[0]):].tolist() 
        text_output = self.tokenizer.decode(output, skip_special_tokens=True)
        return text_output


class QwenModel(HFModels):
    def post_process_response(self, raw_response):
        if self.use_chat_style:
            return raw_response.split("</think>")[-1]
        return raw_response
    
    def format_prompt(self, instruction:str, examples:list[str], input:str) -> str:
        if self.use_chat_style:
            messages = [{"role": "user", "content": input}]
            chat_style_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            return chat_style_input

        if examples:
            return f"{instruction} \n\n Here are some examples: \n {examples} \n\n {input}"
        return f"{instruction} \n\n {input}"

    def get_generation_config(self) -> LLMConfig:
        from transformers import GenerationConfig
        return GenerationConfig(
            max_new_tokens=QwenConfig.max_new_token,
            temperature=QwenConfig.temperature,
            top_k=QwenConfig.top_k,
            top_p=QwenConfig.top_p,
        ) 


class GemmaModel(HFModels):
    def post_process_response(self, raw_response):
        final_response = raw_response.split("<eos>")[0]
        return final_response

    def format_prompt(self, instruction:str, examples:list[str], input:str) -> str:
        if examples:
            return f"{instruction} \n\n Here are some examples: \n {examples} \n\n {input}"
        return f"{instruction} \n\n {input}"

    def get_generation_config(self) -> LLMConfig:
        from transformers import GenerationConfig
        return GenerationConfig(
            max_new_tokens=GemmaConfig.max_new_token,
            temperature=GemmaConfig.temperature,
            top_k=GemmaConfig.top_k,
            top_p=GemmaConfig.top_p,
        ) 