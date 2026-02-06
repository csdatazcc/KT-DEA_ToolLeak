# llms/ollama_llm.py
import ollama
from typing import List, Dict, Any, Union
from llms.base_llm import BaseLLM
from prompt_convert import get_converter
class OllamaLLM(BaseLLM):
    """
    使用 Ollama API 的 LLM 实现。
    确保你的 Ollama 服务正在运行。
    """

    def __init__(self, model: str = "llama2:7b", host: str = None, **kwargs):
        """
        初始化 Ollama LLM。

        Args:
            model (str): 要使用的 Ollama 模型名称 (例如 "llama3", "mistral")。
                        请确保该模型已通过 `ollama pull <model_name>` 拉取到本地。
            host (str, optional): Ollama 服务的主机地址。如果为 None，则使用默认地址。
            **kwargs: 传递给 ollama.Client 的其他初始化参数。
        """
        super().__init__()
        self.model = model
        try:
            # 如果提供了 host，则传递给 Client
            client_args = {}
            if host:
                client_args['host'] = host
            client_args.update(kwargs) # 合并其他参数
            # self.client = ollama.Client(**client_args)
            self.client = ollama.Client(host='')
            # 尝试与 Ollama 服务通信以验证连接和模型可用性
            self._check_model_availability()

        except Exception as e:
            print(f"Error initializing Ollama client or checking model availability: {e}")
            print("Please ensure the Ollama service is running and the model '{self.model}' is available.")
            raise

    def _check_model_availability(self):
        """检查指定的模型是否在 Ollama 中可用。"""
        try:
            available_models = self.client.list()['models']
            model_names = [m['name'] for m in available_models]
            # Ollama 模型名称可能包含 tag，例如 'llama3:latest'
            base_model_name = self.model
            if not any(m.startswith(base_model_name) for m in model_names):
                 raise ValueError(f"Model '{self.model}' not found in available Ollama models: {model_names}. "
                                  "Use 'ollama list' to see available models and 'ollama pull {self.model}' to download.")
        except ConnectionRefusedError:
             raise ConnectionRefusedError("Could not connect to Ollama service. Is it running?")
        except Exception as e:
             print(f"\033[33mWarning: Could not verify model availability due to error: {e}\033[0m")
             # 选择不抛出异常，允许用户在服务启动后重试，但在 generate 时可能会失败

    def generate(self, prompt: Union[str, List[Dict[str, str]]], **kwargs: Any) -> str:
        """
        使用 Ollama API 生成文本。这里使用 ollama.generate API。

        Args:
            prompt (Union[str, List[Dict[str, str]]]): 输入提示。
            **kwargs: 传递给 ollama.generate 的额外参数 (例如 'options': {'temperature': 0.8})。
                      参考 Ollama REST API 文档了解可用选项。

        Returns:
            str: LLM 生成的文本响应。

        Raises:
            ValueError: 如果 prompt 格式无效。
            Exception: Ollama API 调用错误或其他处理错误。
        """
        convert = get_converter("ollama")
        formatted_prompt = convert.to_llm_input(prompt)

        try:
            # 准备传递给 ollama.generate 的参数
            generate_params = {
                "model": self.model,
                "prompt": formatted_prompt,
                "stream": False, # 我们需要完整响应，所以不使用流式传输
            }

            # 合并用户指定的 options
            if 'options' in kwargs:
                generate_params['options'] = kwargs.pop('options')
            if 'system' in kwargs: # 允许传递系统提示词
                 generate_params['system'] = kwargs.pop('system')
            # 其他顶级参数可以根据需要添加，但目前 ollama.generate 主要接受 model, prompt, system, template, context, options, stream, format, keep_alive

            # print(f"DEBUG: Sending to Ollama: {generate_params}") # 调试信息

            response = self.client.generate(**generate_params)

            # 'response' 键包含生成的文本
            content = convert.from_llm_output(response)
            # print(f"DEBUG: Received from Ollama: {content}") # 调试信息
            return content,0

        except Exception as e:
            print(f"An error occurred during Ollama API call: {e}")
            # 检查是否是连接错误
            if "connection refused" in str(e).lower():
                print("Hint: Ensure the Ollama service is running.")
            # 检查是否是模型未找到错误 (虽然构造函数里检查了，但以防万一)
            elif "model not found" in str(e).lower():
                 print(f"Hint: Ensure the model '{self.model}' is available via 'ollama list' or 'ollama pull {self.model}'.")
            raise