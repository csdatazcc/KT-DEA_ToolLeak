# agent_framework/llms/openai_llm.py
import openai
import os
import logging
from typing import List, Dict, Union
from llms.base_llm import BaseLLM
from dotenv import load_dotenv
from prompt_convert import get_converter

# 加载环境变量 (例如 OPENAI_API_KEY)
load_dotenv()

logger = logging.getLogger(__name__)

class OpenAILLM(BaseLLM):
    """
    使用 openai 库与 OpenAI API 交互的 LLM 实现。
    """
    def __init__(self, model: str = "gpt-3.5-turbo", base_url: str = None, api_key: str = None, temperature = 1.0, **kwargs):
        """
        初始化 OpenAI LLM。

        :param model_name: 要使用的 OpenAI 模型名称 (例如 'gpt-4', 'gpt-3.5-turbo')。
        :param api_key: OpenAI API 密钥。如果为 None，则尝试从环境变量 'OPENAI_API_KEY' 获取。
        :param kwargs: 其他传递给 openai.chat.completions.create 的参数 (例如 temperature)。
        """
        super().__init__()
        self.model_name = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature
        if not self.api_key:
            raise ValueError("OpenAI API key not provided or found in environment variable 'OPENAI_API_KEY'")
        self.base_url = base_url
        self.client = openai.OpenAI(base_url = self.base_url, api_key = self.api_key)
        logger.info(f"OpenAI LLM initialized with model: {self.model_name}")

    def generate(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> str:
        """
        使用 OpenAI Chat Completion API 生成文本。

        OpenAI 的 Chat API 推荐使用消息列表格式。如果输入是字符串，会将其包装成用户消息。

        :param prompt: 输入提示。最好是消息列表，例如 [{'role': 'user', 'content': 'Hello'}]。
                       如果是字符串，会被转换为 [{'role': 'user', 'content': prompt}]。
        :param kwargs: 传递给 `client.chat.completions.create` 的额外参数
                       (例如 temperature, max_tokens)。
        :return: LLM 生成的文本响应。
        """
        # 检查输入提示的类型
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

        except openai.APIError as e:
            print(f"OpenAI API error: {e}")
            #logger.error(f"OpenAI API error: {e}")
            # raise e
        except Exception as e:
            logger.error(f"An unexpected error occurred during OpenAI call: {e}")
            raise e
        return generated_text.strip(),tokens