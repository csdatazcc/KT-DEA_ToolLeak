# base_agent.py
"""
该模块定义了Agent的基类 (BaseAgent)。
所有的具体Agent类型都应该继承自这个基类。
"""

from typing import List, Dict, Any
from prompt_convert import  CORE_AGENT_SYSTEM_PROMPT,BASE_SYSTEM_PROMPT
from prompt_convert import get_converter

class BaseAgent:
    """
    Agent的抽象基类。
    定义了Agent的基本结构和规划流程。
    """

    def __init__(self, name: str, llm, tools: Dict[str, Any] = None, system_prompt: str = BASE_SYSTEM_PROMPT):
        """
        初始化BaseAgent。

        Args:
            name (str): Agent的名称。
            llm (Any): 用于生成文本的LLM实例。
            tools (Dict[str, Any], optional): Agent可以使用的工具字典，键为工具名称，值为工具实例。默认为None。
            system_prompt (str): Agent的系统提示词。默认为空字符串。
        """
        self.name = name
        self.llm = llm
        self.tools = tools if tools is not None else []
        self.system_prompt = system_prompt
        self.memory: str=None #用于存储Agent的交互历史
        tool_descriptions_list = []
        if self.system_prompt == BASE_SYSTEM_PROMPT:
            if len(self.tools) != 0:
                for index, tool in enumerate(self.tools):
                    tool_descriptions_list.append(f"- {index+1}. {tool.name}: {tool.description}")
            tool_descriptions_str = "\n".join(tool_descriptions_list)
            self.system_prompt = self.system_prompt.format(tool_descriptions=tool_descriptions_str)
        self.converter = get_converter(self.llm) 
    def add_tool(self, tool_name: str, tool: Any):
        """
        向Agent添加一个工具。

        Args:
            tool_name (str): 工具的名称。
            tool (Any): 工具的实例。
        """
        self.tools[tool_name] = tool

    def get_tool(self ) -> Any:
        """
        获取Agent的工具实例。

        Args:
            tool_name (str): 工具的名称。

        Returns:
            Any: 工具的实例。
        """
        tool_dict = {}
        if len(self.tools) != 0:
            for index, tool in enumerate(self.tools):   
                tool_dict[tool.name] = tool
        return tool
    def _format_tool_descriptions(self) -> str:
        """
        格式化所有可用工具的描述，以便包含在发送给 LLM 的提示中。
        这有助于 LLM 了解它有哪些能力。
        """
        return "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools])
    def _call_llm(self, prompt: str) -> str:
        """
        调用LLM生成响应。

        Args:
            prompt (str): 发送给LLM的提示。

        Returns:
            str: LLM生成的响应。
        """
        return self.llm.generate(prompt)

    def _execute_tool(self, tool_name: str, arguments: str) -> Any:
        """
        执行指定的工具。

        Args:
            tool_name (str): 要执行的工具的名称。
            arguments (Dict[str, str]): 传递给工具的参数。

        Returns:
            str: 工具执行的结果。
        """
        for tool in self.tools:
            if tool.name == tool_name:
                return tool.run(arguments)
        return f"Error: Tool '{tool_name}' not found."

    def plan(self, task: str) -> str:
        """
        Agent的规划方法，具体的规划逻辑由子类实现。

        Args:Initial memory
            task (str): 用户给Agent的任务。

        Returns:
            str: Agent的最终输出或思考过程。
        """
        raise NotImplementedError("Subclasses must implement the plan method.")
    def add_memory(self, memory: str):
        """
        向Agent的内存中添加新的交互历史。

        Args:
            memory (List[str]): 要添加的交互历史。
        """
        if self.memory is None:
            self.memory = memory
        else:
            self.memory+=memory
    def run(self, task: str) -> str:
        """
        执行Agent的主要流程。

        Args:
            task (str): 用户给Agent的任务。

        Returns:
            str: Agent的最终输出。
        """
        input_task = self.converter.to_llm_input({"role":"user","content":task})
        self.memory.append([input_task])
        output = self.plan(input_task)
        self.memory.append(f"{self.name}: {output}")
        return output