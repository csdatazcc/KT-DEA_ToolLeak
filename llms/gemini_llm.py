import os
import logging
from typing import Union, List, Dict, Any
from llms.base_llm import BaseLLM
from dotenv import load_dotenv

from openai import OpenAI

from prompt_convert import get_converter

load_dotenv()
logger = logging.getLogger(__name__)

class GeminiLLM(BaseLLM):
    def __init__(self, model: str = "gemini-2.5-flash", api_key: str = None, base_url: str = None,temperature = 1.0, **kwargs):
        super().__init__()
        self.model_name = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.base_url = base_url or os.getenv("GEMINI_API_BASE_URL", "https://api.openai-proxy.org/google")
        self.temperature = temperature

        if not self.api_key:
            raise ValueError("Gemini API key not provided or found in environment variable 'GEMINI_API_KEY'")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        logger.info(f"Gemini LLM initialized with model: {self.model_name}")

    def generate(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> str:
        # 使用 converter 将内部消息格式转为 Gemini 输入字符串
        convert = get_converter("openai")
        messages = convert.to_llm_input(prompt)
        if self.temperature!=1.0:
            request_params = {
                'model': self.model_name,
                'messages': messages,
            }
        else:
            request_params = {
                'model': self.model_name,
                'messages': messages,
                'temperature': self.temperature,
            }
        if kwargs is not None and not isinstance(kwargs, dict):
            raise ValueError("Kwargs must be of type None or dict!")
        if kwargs:
            for key, value in kwargs.items():
                request_params[key] = value
        logger.debug(f"Generating response with OpenAI using model {self.model_name} and messages.")
        try:
            response = self.client.chat.completions.create(**request_params)
            
            generated_text,tokens= convert.from_llm_output(response)

        except Exception as e:
            logger.error(f"An unexpected error occurred during OpenAI call: {e}")
            raise e
        return generated_text.strip(),tokens
