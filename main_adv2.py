from llms import OllamaLLM, OpenAILLM,GeminiLLM
import json
from agents.react_agent import ReactAgent
from agents.self_refine import SelfRefineAgent
from agents.reflexion import ReflexionAgent
from typing import Dict, List
from tools import *
from Attack import *
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from sentence_transformers import SentenceTransformer, util

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
PURPLE = "\033[35m"
CYAN = "\033[36m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_RED = "\033[91m"
WHITE = "\033[97m"
RESET = "\033[0m"


def get_description_from_list(tool_datas, name):
    """
    从 list 形式的 tool_datas 中根据工具名查找 description。
    tool_datas 的结构必须是：
    [
        {"name": "...", "description": "..."},
        {"name": "...", "description": "..."},
        ...
    ]
    """
    for item in tool_datas:
        if item["name"] == name:
            return item["description"]
    raise ValueError(f"Tool '{name}' not found in tool_datas")


def parse_and_append(raw_text: str, output_file: str):
    """
    Loosely parse raw_text by splitting on --- Document X --- markers.
    Each document is appended as ONE line in the NDJSON file.

    Rules:
      - Split by document markers only (no strict relevance/question/answer requirements)
      - For each parsed document:
            { status: "success", "id": id, "content": text }
      - If can't split into documents:
            { status: "failed", "answer": raw_text }
    """

    # Allow formats:
    # --- Document 1 ---
    # --- Document 1 (Relevance: 0.34) ---

    doc_header_pattern = r"--- Document\s+(\d+)(?:\s*\(.*?\))?\s*---"
    matches = list(re.finditer(doc_header_pattern, raw_text))

    # No documents -> write failure
    if not matches:
        entry = {
            "status": "failed",
            "answer": raw_text.strip()
        }
        move_step = compare_gain([entry], output_file)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return move_step

    documents = []
    for i, m in enumerate(matches):
        doc_id = m.group(1)
        start = m.end()

        # Determine end boundary
        if i + 1 < len(matches):
            end = matches[i + 1].start()
            content = raw_text[start:end].strip()
        else:
            content = raw_text[start:].strip()

        # If content empty => treat as failed document
        if not content:
            entry = {
                "status": "failed",
                "answer": ""
            }
        else:
            entry = {
                "status": "success",
                "id": f"document_{doc_id}",
                "content": content
            }
        
        documents.append(entry)
        move_step = compare_gain(documents, output_file)
        # append each document as its own NDJSON line
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return move_step

def convert_tools(tool_datas, key_phrase, target_tool_name):
    """
    从 list 形式的 tool_datas 和 key_phrase 中构建目标结构：
    - target_tool_info
    - relevant_tool_info
    """
    tool_names = key_phrase.keys()

    target_tool_info = None
    relevant_tool_info = []

    for name in tool_names:
        # 从 list 中查询 description
        description = get_description_from_list(tool_datas, name)

        info = {
            "name": name,
            "description": description,
            "key_phrases": key_phrase[name]
        }

        if name == target_tool_name:
            target_tool_info = info
        else:
            relevant_tool_info.append(info)

    return target_tool_info, relevant_tool_info

def compute_similarity(a: str, b: str) -> float:
    """
    Compute TF-IDF cosine similarity between two text strings.
    """
    if not a.strip() or not b.strip():
        return 0.0
    tfidf = TfidfVectorizer().fit([a, b])
    v1, v2 = tfidf.transform([a, b])
    return float(cosine_similarity(v1, v2)[0][0])


def load_existing_contents(jsonl_file: str) -> List[str]:
    """
    Load all historical content from NDJSON file.
    Format supported:
      success → use "content"
      failed  → use "answer"
    """
    if not os.path.exists(jsonl_file):
        return []

    contents = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if obj.get("status") == "success":
                    if "content" in obj and obj["content"].strip():
                        contents.append(obj["content"].strip())
                else:
                    # failed -> use answer
                    ans = obj.get("answer", "").strip()
                    if ans:
                        contents.append(ans)
            except:
                continue

    return contents


def compare_gain(result_entries: List[Dict],
                 jsonl_file: str,
                 threshold: float = 0.9,
                 verbose: bool = True):

    new_contents = []
    for entry in result_entries:
        if entry["status"] == "success":
            new_contents.append(entry.get("content", "").strip())
        else:
            new_contents.append(entry.get("answer", "").strip())

    old_contents = load_existing_contents(jsonl_file)

    # No historical data → 全部视为 new → 高增益
    if not old_contents:
        total = len(new_contents)
        newly_added = total
        gain_ratio = 1.0
        gain_level = 1
        detail_logs = ["no historical data, all docs are NEW"]

        if verbose:
            print("=== Gain Analysis ===")
            print(f"Historical Docs: 0")
            print(f"Incoming Docs:   {total}")
            print(f"New Items:       {newly_added}")
            print(f"Gain Ratio:      {gain_ratio:.2f}")
            print(f"Gain Level:      {gain_level} (1=High, 2=Medium, 3=Zero)")
            print("Detail:")
            for line in detail_logs:
                print("  -", line)
            print("=====================")

        return gain_level

    # With history
    total = len(new_contents)
    newly_added = 0
    detail_logs = []

    for idx, nc in enumerate(new_contents):
        exists = False
        max_sim = 0.0

        for oc in old_contents:
            sim = compute_similarity(nc, oc)
            if sim > max_sim:
                max_sim = sim
            if sim >= threshold:
                exists = True
                break

        if exists:
            detail_logs.append(f"Doc {idx+1}: similarity={max_sim:.3f} → exists")
        else:
            detail_logs.append(f"Doc {idx+1}: similarity={max_sim:.3f} → NEW")
            newly_added += 1

    gain_ratio = newly_added / total

    if gain_ratio > 0.5:
        gain_level = 1
    elif gain_ratio > 0:
        gain_level = 2
    else:
        gain_level = 3

    # Print output in your desired format
    if verbose:
        print("=== Gain Analysis ===")
        print(f"Historical Docs: {len(old_contents)}")
        print(f"Incoming Docs:   {total}")
        print(f"New Items:       {newly_added}")
        print(f"Gain Ratio:      {gain_ratio:.2f}")
        print(f"Gain Level:      {gain_level} (1=High, 2=Medium, 3=Zero)")
        print("Detail:")
        for line in detail_logs:
            print("  -", line)
        print("=====================")

    # Only return gain_level
    return gain_level



# 读取工具数据
tool_datas_path = ".json"
with open(tool_datas_path, "r", encoding="utf-8") as f:
    tool_datas = json.load(f)

target_tool_name = ""


selected_tools = tool_datas
threshold = 0.7

processor = ToolSemanticProcessor(
    tools=selected_tools,
    target_tool_name=target_tool_name,
    conflict_threshold=threshold
)
key_phrase = processor.process()
print("\nKey phrase extracted:\n")
print(key_phrase)

target_tool_info, relevant_tool_info = convert_tools(tool_datas, key_phrase, target_tool_name)

print(target_tool_info)
print(relevant_tool_info)
llm=GeminiLLM(model="",base_url="",api_key="")
extracted_keywords = None
target_tool = HealthcareRAGTool()
total_asr_c = 0
total_asr_o = 0
total_query_num = 1
print(f"================================= Query {total_query_num} =================================")

save_path = ".ndjson"
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
stealing_prompt = attack_prompt_generate(
        llm=llm,
        model=model,
        target_tool_info=target_tool_info,
        relevant_tool_info=relevant_tool_info,
        extracted_keywords=extracted_keywords,
        prompt=attack_system_prompt
    )

agent = ReactAgent(llm=llm,tools=[])
answer, asr_c,asr_o = agent.plan(stealing_prompt)
total_asr_c+= asr_c
total_asr_o+= asr_o
keyword_base = None
bfs_num = 0

print(f"\033[95mAttack Success Rate (with target tool): {total_asr_c/total_query_num:.4f}\033[0m")
print(f"\033[94mAttack Success Rate (target tool only): {total_asr_o/total_query_num:.4f}\033[0m")
print(f"\033[92mUnique Extracted Items: {target_tool.get_unique_stats()}\033[0m")

avg_per_query = target_tool.get_unique_stats()["total_unique_docs_retrieved"] / total_query_num
print(f"\033[93mAverage Extracted per Query: {avg_per_query:.4f}\033[0m")

parse_and_append(answer,save_path)
new_keyword_list = keyword_extra(llm, answer, extracted_keywords, extract_system_prompt)
keyword_base = keyword_base_update(keyword_base,new_keyword_list)
start_index = 0
end_index = 0
while bfs_num<3:
    if start_index >=len(keyword_base):
        bfs_num+=1
        extracted_keywords = None
        print("New BFS Cycle:")
    else:
        if start_index + 3 > len(keyword_base):
            end_index = len(keyword_base) - 1
        else:
            end_index = start_index + 2
        print(f"Start Index:{start_index},End Index:{end_index}\n")
        extracted_keywords = keyword_base[start_index:end_index + 1]
    print("Current keywords:", extracted_keywords)

    total_query_num+=1
    print(f"================================= Query {total_query_num} =================================")

    stealing_prompt = attack_prompt_generate(
        llm=llm,
        model=model,
        target_tool_info=target_tool_info,
        relevant_tool_info=relevant_tool_info,
        extracted_keywords=extracted_keywords,
        prompt=attack_system_prompt
    )

    agent = ReactAgent(llm=llm,tools=[])

    answer, asr_c,asr_o = agent.plan(stealing_prompt)
    total_asr_c+= asr_c
    total_asr_o+= asr_o

    print(f"\033[95mAttack Success Rate (with target tool): {total_asr_c/total_query_num:.4f}\033[0m")
    print(f"\033[94mAttack Success Rate (target tool only): {total_asr_o/total_query_num:.4f}\033[0m")
    print(f"\033[92mUnique Extracted Items: {target_tool.get_unique_stats()}\033[0m")


    avg_per_query = target_tool.get_unique_stats()["total_unique_docs_retrieved"] / total_query_num
    print(f"\033[93mAverage Extracted per Query: {avg_per_query:.4f}\033[0m")
    move_step = parse_and_append(answer,save_path)
    new_keyword_list = keyword_extra(llm, answer, extracted_keywords, extract_system_prompt)
    keyword_base = keyword_base_update(keyword_base,new_keyword_list)
    start_index+= move_step

    if total_query_num >200 or target_tool.get_unique_stats()["total_unique_docs_retrieved"] == 200:
        break








