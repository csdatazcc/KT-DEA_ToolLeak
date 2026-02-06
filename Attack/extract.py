from llms import OllamaLLM
from llms import OpenAILLM
from llms import DeepseekLLM
import json
import re

extract_system_prompt = '''
# Role
You are an expert Data Structuring Analyst. Your task is to extract core information from unstructured text and format the output based on the presence of a "Important Reference Keyword List".

# Input Data
1. {text}: The raw text to be analyzed.
2. {keywords}: A list of reference categories/keywords (can be empty).

# Process
1. **Extraction Phase (Independent & Comprehensive)**:
   - **First**, ignore the {keywords} and analyze {text} deeply.
   - Don't consider the extraction of terms related to tools.
   - Keywords should be unique and significant enough to avoid generic words, while ensuring they have certain semantic value (that is, certain scenarios or semantics can be analyzed from the words).
   - Extract **multiple** critical entities (names, organizations, products), metadata, and core topics.
   - **Requirement**: Do not limit yourself to one term. Extract all terms that are structurally significant to the text.

2. **Logic & Output**:
   Check if {keywords} is provided/non-empty.

   - **Branch A: If {keywords} is PROVIDED**
     - Iterate through every "newly extracted term".
     - Find the semantically closest match from {keywords} for each extracted term.
     - **Aggregation**: Group all extracted terms that map to the same Reference Keyword into a single list.
     - **Format**: Output a JSON list of objects, where each object contains one Reference Keyword pointing to a list of matched terms:
       `[{"Reference_Keyword": ["Extracted_Term_1", "Extracted_Term_2"]}, ...]`

   - **Branch B: If {keywords} is EMPTY**
     - Analyze the content and extract the core terms representing the main topics.
     - Since no specific keywords were provided, limit the output to **the top 5 most critical terms**.
     - **Format**: Output a strict JSON list of strings:
       `["Extracted_Term_1", "Extracted_Term_2", ...]`

# Constraints
- **Semantic Aggregation**: Even if an extracted term is not literally related, force the closest semantic association from the list.
- **Strict JSON**: Output ONLY valid JSON string. No Markdown, no explanations.

# Execution
Start processing:
Text: {{ text }}
Keywords: {{ keywords }}
'''

import json
import ast

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
            # raise ValueError(f"解析失败（在提取主体后）: {e}")
            return 3


def keyword_extra(llm, text, keywords, prompt):

    # ----- Step 1: Prepare prompt -----
    final_prompt = prompt.replace("{{ text }}", str(text))
    final_prompt = final_prompt.replace("{{ keywords }}", str(keywords))

    # ----- Step 2: Generate response -----
    response = llm.generate(final_prompt)[0]

    # ----- Step 3: Parse structured content -----
    parsed = parse_to_dict(response)

    print("[INFO] Parsed result as Python object:")
    print(json.dumps(parsed, indent=2, ensure_ascii=False))
    print("=======================================\n")

    return parsed

def keyword_base_update(keyword_base, new_keyword_list):
    print(f"[INFO] Initial keyword_base: {keyword_base}")
    print(f"[INFO] new_keyword_list received: {new_keyword_list}")

    # If keyword_base is None → initialize as empty list
    if keyword_base is None:
        print("[INFO] keyword_base is None. Initializing as empty list.")
        keyword_base = []

    # Convert dict → list of dicts
    if isinstance(new_keyword_list, dict):
        print("[DEBUG] new_keyword_list is a dict. Converting to list of dicts.")
        new_keyword_list = [new_keyword_list]

    # Case 1: list of dicts
    if isinstance(new_keyword_list, list) and new_keyword_list and isinstance(new_keyword_list[0], dict):
        print("[INFO] Detected dict-based keyword insertion.")

        for d in new_keyword_list:
            for key, values in d.items():
                print(f"\n[DEBUG] Processing key '{key}' with values: {values}")
                num = 0
                if key in keyword_base:
                    pos = keyword_base.index(key)
                    print(f"[DEBUG] Key '{key}' found in keyword_base at position {pos}.")

                    # Remove values already in keyword_base
                    filtered_values = [v for v in values if v not in keyword_base]
                    print(f"[DEBUG] Filtered values to insert after '{key}': {filtered_values}")

                    # Insert after the key
                    num = 0
                    for offset, v in enumerate(filtered_values, 1):
                        keyword_base.insert(pos + offset, v)
                        num+=1
                        print(f"[TRACE] Inserted '{v}' at position {pos + offset}")
                        if num>=5:
                            break

                else:
                    print(f"[DEBUG] Key '{key}' not found. Appending its values to the end.")
                    try:
                        filtered_values = [v for v in values if v not in keyword_base][:3]
                    except:
                        continue
                    print(f"[DEBUG] Filtered append values: {filtered_values}")
                    keyword_base.extend(filtered_values)

    # Case 2: list of strings
    elif isinstance(new_keyword_list, list):
        print("[INFO] Detected simple list of keywords.")
        num = 0
        for kw in new_keyword_list:
            if kw not in keyword_base:
                keyword_base.append(kw)
                num+=1
                print(f"[TRACE] Appended new keyword: {kw}")
                if num>=5:
                    break
            else:
                print(f"[TRACE] Skipped existing keyword: {kw}")

    print(f"[INFO] Final updated keyword_base: {keyword_base}\n")
    return keyword_base