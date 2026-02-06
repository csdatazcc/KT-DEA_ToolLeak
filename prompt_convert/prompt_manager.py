from typing import List, Dict, Optional
from .data_converter import BaseDataConverter, InternalMessageFormat

class PromptManager:
    """
    管理 Agent 的提示构建过程。
    """
    def __init__(self, system_prompt: str, data_converter: BaseDataConverter):
        """
        初始化 PromptManager。

        Args:
            system_prompt (str): 要使用的核心系统提示词文本。
            data_converter (BaseDataConverter): 用于格式化最终输入的数据转换器实例。
        """
        if not system_prompt:
            raise ValueError("System prompt cannot be empty.")
        if not data_converter:
             raise ValueError("Data converter instance is required.")

        self.system_prompt_content = system_prompt.strip()
        self.data_converter = data_converter
        print("[PromptManager]: Initialized with Data Converter:", type(data_converter).__name__)

    def _build_message_history(self,
                               user_input: str,
                               history: Optional[InternalMessageFormat] = None) -> InternalMessageFormat:
        """
        构建包含系统提示、历史记录和当前用户输入的完整消息列表。

        Args:
            user_input (str): 最新的用户输入。
            history (Optional[InternalMessageFormat]): 可选的，之前的对话历史。
                                                       格式为 [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]

        Returns:
            InternalMessageFormat: 构建好的完整消息列表 (Agent 内部标准格式)。
        """
        print("[PromptManager]: Building message history...")

        # 1. 添加系统提示词
        messages: InternalMessageFormat = [{"role": "system", "content": self.system_prompt_content}]
        print(f"[PromptManager]: Added system prompt (length: {len(self.system_prompt_content)} chars).")

        # 2. 添加历史记录 (如果提供)
        if history:
            # 可以在这里添加历史记录截断逻辑，防止超出 LLM 的上下文长度限制
            messages.extend(history)
            print(f"[PromptManager]: Added {len(history)} messages from history.")

        # 3. 添加当前用户输入 (使用 DataConverter 标准化)
        standardized_user_message = self.data_converter.format_user_input(user_input)
        messages.append(standardized_user_message)
        print(f"[PromptManager]: Added current user input: {standardized_user_message}")

        print(f"[PromptManager]: Total messages built: {len(messages)}")
        return messages

    def construct_llm_input(self,
                            user_input: str,
                            history: Optional[InternalMessageFormat] = None,
                            **kwargs) -> Any: # 返回类型取决于转换器
        """
        构建最终要发送给 LLM 的输入。

        Args:
            user_input (str): 最新的用户输入。
            history (Optional[InternalMessageFormat]): 可选的对话历史。
            **kwargs: 传递给 data_converter.to_llm_input 的额外参数。

        Returns:
            Any: 由 DataConverter 生成的、适合 LLM 的输入格式。
        """
        print("[PromptManager]: Constructing final LLM input...")
        # 1. 构建内部标准消息列表
        internal_messages = self._build_message_history(user_input, history)

        # 2. 使用 DataConverter 将内部格式转换为 LLM 特定格式
        llm_input_data = self.data_converter.to_llm_input(internal_messages, **kwargs)
        print("[PromptManager]: LLM input construction complete.")
        return llm_input_data