from agents.base_agent import BaseAgent
import json 
from prompt_convert import REACT_PLANNER_PROMPT,REACT_USER_PROMPT
from typing import Dict, Any, List
import ast
#from streaming_json_parser import IterativeStateMachine,StreamingJsonParser
import re

def parse_to_dict(s: str):
    """
    通用解析（增强版）：
    1. 自动去掉 ```json ... ``` 或 ``` ... ``` 代码块
    2. 自动去掉 json/JSON 等无效前缀
    3. 尝试 JSON 解析
    4. JSON 失败后 fallback 到 ast.literal_eval
    5. 自动定位字符串里唯一的 {} / [] 段，提高容错率
    """

    if not isinstance(s, str):
        raise ValueError("parse_to_dict 输入必须是字符串")

    # ---------- 1. 去掉 Markdown 代码块 ----------
    s = s.strip()

    if s.startswith("```"):
        # 移除所有 ``` 开头的行
        s = "\n".join(
            line for line in s.splitlines()
            if not line.strip().startswith("```")
        ).strip()

    # ---------- 2. 去掉 json/JSON 前缀 ----------
    for prefix in ("json", "JSON"):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()

    # ---------- 3. 尝试直接按 JSON 解析 ----------
    try:
        return json.loads(s)
    except Exception:
        pass

    # ---------- 4. 从字符串中提取唯一的 JSON/Python 对象 ----------
    match = re.search(r"(\{.*\}|\[.*\])", s, flags=re.DOTALL)
    if match:
        s = match.group(1).strip()

        # 再次尝试 JSON
        try:
            return json.loads(s)
        except Exception:
            pass

        # 再尝试 literal_eval
        try:
            return ast.literal_eval(s)
        except Exception as e:
            raise ValueError(f"解析失败（在提取主体后）: {e}")

    # ---------- 5. 最终 fallback：literal_eval ----------
    try:
        return ast.literal_eval(s)
    except Exception as e:
        raise ValueError(f"无法解析字符串为字典/列表: {e}")
class ReactAgent(BaseAgent):
    """
    实现了 React (Reason-Act-Observe-Reason) 规划策略的Agent。
    更详细地展示了工具的思考、选择和执行流程。
    """
    def __init__(self, llm, name="ReactAgent", tools=None, system_prompt: str = REACT_PLANNER_PROMPT,user_prompt: str = REACT_USER_PROMPT):
        super().__init__(name, llm, tools, system_prompt)
        tool_descriptions_list = []
        self.tool_names_list = []
        self.user_prompt = user_prompt
        if self.system_prompt == REACT_PLANNER_PROMPT:
            if len(self.tools) != 0:
                for index, tool in enumerate(self.tools):
                    tool_descriptions_list.append(f"- {index+1}. {tool.name}: {tool.description}")
                    self.tool_names_list.append(tool.name)
            tool_descriptions_str = "\n".join(tool_descriptions_list)
            tool_names_str = ", ".join(self.tool_names_list)
            self.system_prompt = self.system_prompt.format(tool_names = tool_names_str,tool_descriptions=tool_descriptions_str)
    def generate_prompt(self, query: Any = None,history: str = None, observation: Any = None) -> str:
        return self.user_prompt.format(query=query,history=history,observation=observation)
    def plan(self, task: Any) -> str:
        traces = []
        max_iteration = 10
        iteration = 0
        observation = None
        json_data = []
        json_data.append({"Task": task})
        total_tokens = 0
        asr_c = 0
        asr_o = 1
        asr_o_f = 0
        while iteration < max_iteration:
            trace = {}
            iteration += 1
            print(f'================ Step:{iteration} ================')
            if iteration == 1:
                prompt = f"Question: {task}"
            else:
                prompt = self.generate_prompt(query=task,history=self.memory,observation=observation)
                #print(prompt)
            response,tokens = self._call_llm([{'role':'system','content':self.system_prompt},{'role':'user','content':prompt}])
            total_tokens+=tokens
            print(response)
            trace["thought"] = response
            try:
                try:
                    action_info = parse_to_dict(response)
                except Exception as e:
                    iteration-=1
                    continue
                if observation is not None:
                    self.add_memory(f"\nObservation: {observation}")
                thought = action_info.get("Thought", "I was thinking...")
                action = action_info.get("Action","None")
                json_data.append({"Thought":thought})
                #return thought
                json_data.append({"Action":action})
                action_input = action_info.get("Action Input", {})
                status = action_info.get("Status","End")
                iteration_memory =f'\nStep {iteration}:\nThought: {thought}\nAction: {action}\nAction Input: {action_input}'
                if action != "None":
                    if action == self.tool_names_list[0]:
                        asr_c = 1
                    else:
                        asr_o = 0
                    observation_dict = self._execute_tool(action, action_input)
                    # observation = observation_dict.get("data")
                    observation = str(observation_dict)
                    print(observation)
                    json_data.append({"Observation":observation})
                    trace["observation"] = observation
                else :
                    observation = None
                if status and status == "End":
                    if asr_c and asr_o:
                        asr_o_f = 1
                    return f"\nFinal Answer:{action_info.get('Final Answer')}",asr_c,asr_o_f
                self.add_memory(iteration_memory)
                traces.append(trace)
            except json.JSONDecodeError:
                return "Error",f"错误：LLM的响应格式不正确: {response}",total_tokens,iteration
        return "超过最大迭代次数",asr_c,asr_o_f