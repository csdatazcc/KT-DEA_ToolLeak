from agents.base_agent import BaseAgent
import json 
from prompt_convert import RAG_USER_PROMPT,RAG_INITIAL_PROMPT
from typing import Dict, Any, List
from streaming_json_parser import IterativeStateMachine,StreamingJsonParser
from RAG import find_best_match_knowledge

class RAGAgent(BaseAgent):
    """
    实现了 React (Reason-Act-Observe-Reason) 规划策略的Agent。
    更详细地展示了工具的思考、选择和执行流程。
    """
    def __init__(self, llm, name="ReactAgent", tools=None, system_prompt: str = RAG_INITIAL_PROMPT,user_prompt: str = RAG_USER_PROMPT):
        super().__init__(name, llm, tools, system_prompt)
        tool_descriptions_list = []
        tool_names_list = []
        self.user_prompt = user_prompt
        if self.system_prompt == RAG_INITIAL_PROMPT:
            if len(self.tools) != 0:
                for index, tool in enumerate(self.tools):
                    tool_descriptions_list.append(f"- {index+1}. {tool.name}: {tool.description}")
                    tool_names_list.append(tool.name)
            tool_descriptions_str = "\n".join(tool_descriptions_list)
            tool_names_str = ", ".join(tool_names_list)
            self.system_prompt = self.system_prompt.format(tool_names = tool_names_str,tool_descriptions=tool_descriptions_str)
    def generate_prompt(self, query: Any = None,history: str = None, observation: Any = None, case: Any=None) -> str:
        return self.user_prompt.format(query=query,case=case,history=history,observation=observation)
    def plan(self, task: Any) -> str:
        traces = []
        max_iteration = 15
        iteration = 0
        observation = None
        case = find_best_match_knowledge(task)[0]["full_details"]
        json_data = []
        json_data.append({"Task": task})
        total_tokens = 0
        while iteration < max_iteration:
            trace = {}
            iteration += 1
            print(f'================ Step:{iteration} ================')
            if iteration == 1:
                prompt = f"Question: {task}"
            else:
                prompt = self.generate_prompt(query=task,case=case,history=self.memory,observation=observation)
            response,tokens = self._call_llm([{'role':'system','content':self.system_prompt},{'role':'user','content':prompt}])
            total_tokens+=tokens
            if observation is not None:
                self.add_memory(f"\nObservation: {observation}")
            print(response)
            trace["thought"] = response
            try:
                action_info = json.loads(response)
                thought = action_info.get("Thought", "I was thinking...")
                #return thought
                action = action_info.get("Action","None")
                json_data.append({"Thought":thought})
                json_data.append({"Action":action})
                action_input = action_info.get("Action Input", {})
                status = action_info.get("Status","End")
                iteration_memory =f'\nStep {iteration}:\nThought: {thought}\nAction: {action}\nAction Input: {action_input}'
                if action != "None":
                    observation_dict = self._execute_tool(action, action_input)
                    observation = observation_dict.get("data")
                    observation = str(observation)
                    json_data.append({"Observation":observation})
                    trace["observation"] = observation
                else :
                    observation = None
                if status and status == "End":
                    return traces,f"\nFinal Answer:{action_info.get('Final Answer')}",total_tokens,iteration
                self.add_memory(iteration_memory)
                traces.append(trace)
            except json.JSONDecodeError:
                return "Error",f"错误：LLM的响应格式不正确: {response}",total_tokens,iteration
        return "Error","超过最大迭代次数",total_tokens,25
