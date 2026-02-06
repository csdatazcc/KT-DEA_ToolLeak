import openai
import os
import logging
from typing import List, Dict, Union
from llms.base_llm import BaseLLM
from dotenv import load_dotenv
from prompt_convert import get_converter

# 加载环境变量，例如 DEEPSEEK_API_KEY
load_dotenv()

logger = logging.getLogger(__name__)

class DeepseekLLM(BaseLLM):
    """
    使用 OpenAI 兼容接口调用 Deepseek Reasoner 模型的 LLM 实现。
    """
    def __init__(self, model: str = "deepseek-reasoner", base_url: str = None, api_key: str = None, temperature=1.0, **kwargs):
        """
        初始化 Deepseek LLM。

        :param model: 模型名称，默认为 'deepseek-reasoner'。
        :param base_url: API 地址，默认为 None，优先从环境变量 DEEPSEEK_API_BASE_URL 读取。
        :param api_key: API 密钥，优先从环境变量 DEEPSEEK_API_KEY 读取。
        :param temperature: 控制输出的多样性。
        """
        super().__init__()
        self.model_name = model
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.temperature = temperature
        self.base_url = base_url or os.getenv("DEEPSEEK_API_BASE_URL", "https://api.openai-proxy.org/v1")

        if not self.api_key:
            raise ValueError("Deepseek API key not provided or found in environment variable 'DEEPSEEK_API_KEY'")

        self.client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)
        logger.info(f"Deepseek LLM initialized with model: {self.model_name}")

    def generate(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> str:
        """
        使用 Deepseek Reasoner 生成文本。

        :param prompt: 输入提示，支持字符串或消息列表格式。
        :param kwargs: 其他传递给 client.chat.completions.create 的参数。
        :return: 生成的文本和 token 数。
        """
        convert = get_converter("openai")  # Deepseek 的接口与 OpenAI 兼容，复用 "openai" 转换
        messages = convert.to_llm_input(prompt)

        request_params = {
            'model': self.model_name,
            'messages': messages,
            'temperature': self.temperature,
        }

        if kwargs:
            if not isinstance(kwargs, dict):
                raise ValueError("Kwargs must be of type dict!")
            request_params.update(kwargs)

        logger.debug(f"Generating response with Deepseek using model {self.model_name} and messages.")

        try:
            response = self.client.chat.completions.create(**request_params)
            generated_text, tokens = convert.from_llm_output(response)

        except openai.APIError as e:
            logger.error(f"Deepseek API error: {e}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error during Deepseek call: {e}")
            raise e

        return generated_text.strip(), tokens
