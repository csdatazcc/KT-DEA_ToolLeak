from abc import ABC, abstractmethod

class BaseTool(ABC):
    """
    所有工具类的抽象基类。
    每个工具都需要定义名称、描述以及执行方法。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """工具的唯一名称。Agent 将使用此名称来选择工具。"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        工具功能的清晰描述。
        这对 Agent 至关重要，因为它会根据描述来判断何时使用该工具。
        描述应清楚说明工具的作用以及它期望的输入。
        """
        pass

    @abstractmethod
    def run(self, action_input: str) -> str:
        """
        执行工具的核心逻辑。
        :param action_input: 从 Agent 接收到的、执行该工具所需的输入字符串。
        :return: 工具执行结果的字符串表示。
        """
        pass