from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Tuple
# 定义 Agent 内部使用的标准化消息格式
# 通常是一个包含 'role' 和 'content' 的字典列表
# 例如: [{'role': 'system', 'content': '...'}, {'role': 'user', 'content': '...'}]
InternalMessageFormat = Any
from google.genai import types

# 定义 LLM 可能的输入输出格式 (这里用 Any，具体实现会更精确)
LlmInputFormat = Any
LlmOutputFormat = Any


class BaseDataConverter(ABC):
    """
    数据转换器抽象基类。
    所有具体的 LLM 数据转换器都应继承此类并实现其抽象方法。
    """

    @abstractmethod
    def to_llm_input(self, messages: InternalMessageFormat, **kwargs) -> LlmInputFormat:
        """
        将 Agent 内部标准消息格式转换为特定 LLM 的输入格式。

        Args:
            messages (InternalMessageFormat): Agent 内部标准格式的消息列表。
            **kwargs: 可能需要的额外参数 (例如，模型特定的配置)。

        Returns:
            LlmInputFormat: 适用于目标 LLM API 的输入格式。
        """
        pass

    @abstractmethod
    def from_llm_output(self, llm_output: LlmOutputFormat, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        将特定 LLM 的原始输出转换为 Agent 内部标准响应格式 (通常是助手的文本回复)。

        Args:
            llm_output (LlmOutputFormat): LLM API 返回的原始输出。
            **kwargs: 可能需要的额外参数。

        Returns:
            Tuple[str, Dict[str, Any]]:
                - str: 从 LLM 输出中提取的助手回复文本。
                - Dict[str, Any]: 可能包含的其他元数据 (例如，token 使用情况, finish reason 等)。
        """
        pass

    def format_user_input(self, user_input: str) -> Dict[str, str]:
        """
        将原始用户输入字符串标准化为内部消息字典格式。
        这是一个通用方法，但如果需要，子类可以覆盖它。

        Args:
            user_input (str): 用户的原始输入文本。

        Returns:
            Dict[str, str]: 标准化的用户消息字典。
        """
        pass
        return {"role": "user", "content": user_input.strip()}

    def format_assistant_output(self, assistant_response: str) -> Dict[str, str]:
        """
        将助手的回复文本标准化为内部消息字典格式。

        Args:
            assistant_response (str): 助手的回复文本。

        Returns:
            Dict[str, str]: 标准化的助手消息字典。
        """
        pass
        return {"role": "assistant", "content": assistant_response.strip()}


# --- 具体实现示例 ---

class OpenAIConverter(BaseDataConverter):
    """
    适用于 OpenAI Chat Completion API (gpt-3.5-turbo, gpt-4 等) 的数据转换器。
    输入: List[Dict[str, str]] (与内部格式一致)
    输出: OpenAI API 的 JSON 响应对象
    """

    def to_llm_input(self, prompt: InternalMessageFormat, **kwargs) -> LlmInputFormat:
        """
        对于 OpenAI，内部格式通常与其 API 格式兼容。
        可以直接返回消息列表，或者根据需要添加其他 API 参数。

        Args:
            messages (InternalMessageFormat): 标准消息列表。
            **kwargs: 传递给 OpenAI API 的其他参数 (如 model, temperature 等)。

        Returns:
            Dict[str, Any]: 包含 'messages' 键和其他 API 参数的字典。
                           或者仅仅是 messages 列表，取决于调用方式。
                           这里我们返回完整的参数字典。
        """
        if not isinstance(prompt, (str, list)):
            raise TypeError("Prompt must be a string or list of message dicts")
        if isinstance(prompt, list):
            if not all(isinstance(msg, dict) and 'role' in msg and 'content' in msg for msg in prompt):
                raise ValueError("Prompt list must contain dictionaries with 'role' and 'content' keys")
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            raise TypeError("Prompt must be a string or list of message dicts")
        return messages

    def from_llm_output(self, llm_output: LlmOutputFormat, **kwargs) -> str:
        """
        从 OpenAI API 的响应中提取助手消息内容。

        Args:
            llm_output (LlmOutputFormat): 假设是 OpenAI API 返回的 JSON 对象 (已解析为字典)。
                                         例如: {'choices': [{'message': {'role': 'assistant', 'content': '...'}, 'finish_reason': 'stop'}], 'usage': {...}}
            **kwargs: 备用参数。

        Returns:
            Tuple[str, Dict[str, Any]]:
                - str: 助手的回复文本。
                - Dict[str, Any]: 包含 'usage' 和 'finish_reason' 等元数据。
        """
        try:
            # 健壮性检查
            generated_text = ""
            if llm_output.choices and len(llm_output.choices) > 0:
                message_content = llm_output.choices[0].message.content
                tokens = llm_output.usage.total_tokens
                if message_content is not None:
                     generated_text = message_content
                else:
                    # 处理 function call 或其他非文本响应 (如果未来需要)
                    generated_text = "" # 或者可以返回响应对象的其他部分
                    tokens = 0
            else:
                generated_text = ""
                tokens = 0
            return generated_text.strip(),tokens
        except (TypeError, IndexError, KeyError) as e:
            print(f"[OpenAIConverter]: Error parsing LLM output: {e}. Output was: {llm_output}")
            # 返回错误信息或默认值，避免程序崩溃
            return "[Error: Failed to parse LLM response]", {"error": str(e)}
    def format_user_input(self, user_input: str) -> Dict[str, str]:
        """
        Args:
            user_input (str): 用户的原始输入文本。

        Returns:
            Dict[str, str]: 标准化的用户消息字典。
        """
        pass
        return {"role": "user", "content": user_input.strip()}

    def format_assistant_output(self, assistant_response: str) -> Dict[str, str]:
        """
        将助手的回复文本标准化为内部消息字典格式。

        Args:
            assistant_response (str): 助手的回复文本。

        Returns:
            Dict[str, str]: 标准化的助手消息字典。
        """
        pass
        return {"role": "assistant", "content": assistant_response.strip()}

class OllamaConverter(BaseDataConverter):
    def to_llm_input(self, prompt: InternalMessageFormat, **kwargs) -> LlmInputFormat:
        """
        将输入 prompt 转换为 Ollama API (chat) 通常接受的单个字符串格式。
        对于消息列表，将其简单地连接起来。
        注意：Ollama 的 chat API 也接受消息列表，但 generate API 通常接受字符串。
             这里为了与 generate API 保持一致性，先转换为字符串。
             如果使用 chat API，可以调整此方法或 generate 方法。
        """
        if isinstance(prompt, str):
            return prompt
        elif isinstance(prompt, list):
            # 将消息列表格式化为简单的文本对话历史
            formatted_prompt = ""
            for message in prompt:
                try:
                    if 'role' not in message or 'content' not in message:
                        raise KeyError("The dictionary must contain the 'role' and 'content' keys")
                except KeyError as e:
                    print(f"Error formatting message: {e}")
                role = message.get("role", "user") # 默认为 user
                content = message.get("content", "")
                formatted_prompt += f"{role.capitalize()}: {content}\n\n"
            return formatted_prompt.strip()
        elif isinstance(prompt, dict):
            formatted_prompt = ""
            try:
                if 'role' not in prompt or 'content' not in prompt:
                    raise KeyError("The dictionary must contain the 'role' and 'content' keys")
            except KeyError as e:
                print(f"Error formatting message: {e}")
            role = prompt.get("role", "user") # 默认为 user
            content = prompt.get("content", "")
            formatted_prompt += f"{role.capitalize()}: {content}\n\n"
            return formatted_prompt.strip()
        else:
            raise TypeError("Prompt must be a string or a list of dictionaries.")


    def from_llm_output(self, llm_output: LlmOutputFormat, **kwargs) -> str:
        content = llm_output.get('response', '')
        return content.strip()
    def format_user_input(self, user_input: str) -> Dict[str, str]:
        """
        将原始用户输入字符串标准化为内部消息字典格式。
        这是一个通用方法，但如果需要，子类可以覆盖它。

        Args:
            user_input (str): 用户的原始输入文本。

        Returns:
            Dict[str, str]: 标准化的用户消息字典。
        """
        pass
        return {"role": "user", "content": user_input.strip()}

    def format_assistant_output(self, assistant_response: str) -> Dict[str, str]:
        """
        将助手的回复文本标准化为内部消息字典格式。

        Args:
            assistant_response (str): 助手的回复文本。

        Returns:
            Dict[str, str]: 标准化的助手消息字典。
        """
        pass
        return {"role": "assistant", "content": assistant_response.strip()}

class GeminiConverter(BaseDataConverter):

    def to_llm_input(self, messages, **kwargs):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages.strip()}]

        system_instruction = None
        normal_messages = []

        # 1. 分离 System Prompt
        for msg in messages:
            if msg.get("role") == "system":
                # 如果有多个 system，拼接起来
                current_sys = msg.get("content", "")
                if system_instruction:
                    system_instruction += "\n" + current_sys
                else:
                    system_instruction = current_sys
            else:
                normal_messages.append(msg)

        # 2. 构建 Gemini 格式的消息列表
        contents = []
        for msg in normal_messages:
            role = msg["role"]
            # 映射 role: assistant -> model
            if role == "assistant":
                role = "model"
            
            # 确保 role 只有 user 或 model
            if role not in ["user", "model"]:
                # 或者是抛出错误，或者强制转为 user
                role = "user" 

            # 使用 SDK 的 types 构建，比纯 dict 更安全
            contents.append(types.Content(
                role=role,
                parts=[types.Part(text=msg["content"])]
            ))

        # 3. 将 system_instruction 返回给调用者，或者在这里处理
        # 注意：你需要修改调用处的代码来接收 system_instruction
        return contents, system_instruction



    def from_llm_output(self, llm_output: LlmOutputFormat, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        将 Gemini API 输出转换为内部格式
        """
        text = ""
        metadata = {}

        try:
            if hasattr(llm_output, "candidates") and llm_output.candidates:
                cand = llm_output.candidates[0]
                if hasattr(cand, "content") and hasattr(cand.content, "parts"):
                    for p in cand.content.parts:
                        if hasattr(p, "text") and p.text:
                            text += p.text
                        elif hasattr(p, "raw_text") and p.raw_text:
                            text += p.raw_text

            if hasattr(llm_output, "usage_metadata"):
                md = llm_output.usage_metadata
                for field in ["prompt_token_count", "candidates_token_count", "total_token_count"]:
                    if hasattr(md, field):
                        metadata[field] = getattr(md, field)

            return text.strip(), metadata

        except Exception as e:
            return "[Error parsing Gemini output]", {"error": str(e)}


def get_converter(llm_type: Any) -> BaseDataConverter:
    """
    根据 LLM 类型获取对应的数据转换器实例。

    Args:
        llm_type (str): LLM 的类型标识符 (例如, 'openai', 'chatml').

    Returns:
        BaseDataConverter: 对应的数据转换器实例。

    Raises:
        ValueError: 如果找不到指定类型的转换器。
    """
    from llms import OllamaLLM,OpenAILLM,GeminiLLM,DeepseekLLM
    if isinstance(llm_type, str):
        llm_type_lower = llm_type.lower()
        if llm_type_lower == 'openai':
            return OpenAIConverter()
        elif llm_type_lower == 'ollama':
            return OllamaConverter()
        elif llm_type_lower == 'gemini':
            return GeminiConverter()
        elif llm_type_lower == "deepseek":
            return OpenAIConverter()
    elif isinstance(llm_type, OpenAILLM):
        return OpenAIConverter()
    elif isinstance(llm_type, OllamaLLM):
        return OllamaConverter()
    elif isinstance(llm_type,GeminiLLM):
        return GeminiConverter()
    elif isinstance(llm_type,DeepseekLLM):
        return OpenAIConverter()
    else:
        raise ValueError(f"Unsupported LLM type for data conversion: {llm_type}")
    