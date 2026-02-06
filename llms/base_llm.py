from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union

class BaseLLM(ABC):
    """
    LLM 接口的抽象基类。
    所有具体的 LLM 实现都应继承此类并实现 generate 方法。
    """

    @abstractmethod
    def generate(self, prompt: Union[str, List[Dict[str, str]]], **kwargs: Any) -> str:
        """
        生成文本的核心方法。

        Args:
            prompt (Union[str, List[Dict[str, str]]]):
                输入的提示。可以是单个字符串，也可以是符合特定模型要求的消息列表
                （例如 OpenAI 的 [{"role": "user", "content": "..."}] 格式）。
                具体的实现类需要处理这两种情况或说明其支持的格式。
            **kwargs: 传递给底层 LLM API 的额外参数 (例如 temperature, max_tokens 等)。

        Returns:
            str: LLM 生成的文本响应。

        Raises:
            NotImplementedError: 如果子类没有实现此方法。
            Exception: 可能在 API 调用或处理过程中引发其他异常。
        """
        raise NotImplementedError

    def __call__(self, prompt: Union[str, List[Dict[str, str]]], **kwargs: Any) -> str:
        """
        允许将 LLM 实例像函数一样调用。
        """
        return self.generate(prompt, **kwargs)
