from llms import *
import json
from sentence_transformers import SentenceTransformer, util


Tool_System_Prompt= '''
    # Role
You are a professional synthetic data generation expert specializing in creating high-quality "Knowledge Retrieval Tools".

# Goal
Based on the provided [Target Tool Description] and an integer [K], you need to generate K related synthetic tools.

# Constraints & Workflow
1.  **Semantic Similarity (to Target Tool)**:
    *   Every generated tool must strictly belong to the same specific domain as the [Target Tool] (e.g., Finance, Medical, Coding).
    *   The generated tools must be "Knowledge-based," meaning their primary function is to retrieve data, query documents, fetch context, or access historical records.

2.  **Semantic Distinctness (between Synthetic Tools)**:
    *   The functionality of the K generated tools must be as orthogonal or mutually exclusive as possible.
    *   Avoid functional overlap (e.g., do not generate both "Stock Price Query" and "Share Value Retriever").
    *   You need to decompose the target domain into K distinct sub-dimensions (e.g., Real-time Data, Historical Archives, Sentiment Analysis, Regulations, Basic Encyclopedia, etc.).

3.  **Output Format**:
    *   Output a strictly formatted JSON List (Array) containing K objects.
    *   Each object must contain `name` and `description` fields.
    *   `name` must be in CamelCase and end with "Tool".
    *   `description` must detail the specific utility, expected input, and the specific aspect of the domain it addresses.

# Input Format
Target Tool: {Description}
K: {Integer}

# Output Example
[
    {
        "name": "SpecificSubDomainTool",
        "description": "Useful for [specific function]. Input should be [specific input]."
    },
    ...
]
'''
def cosine_similarity(a, b):
    return util.cos_sim(a, b).item()
import json
import os

def append_tools_to_json(tools, json_path):
    """
    Append a list of tool dicts to a JSON file.
    Ensures the file is always a JSON array.
    """

    # Case: file does not exist → create with tools
    if not os.path.exists(json_path):
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(tools, f, ensure_ascii=False, indent=4)
        return

    # Case: file exists → read old content
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            existing_data = json.load(f)
            if not isinstance(existing_data, list):
                raise ValueError("JSON root must be a list.")
        except json.JSONDecodeError:
            # If file corrupted or empty → reset as list
            existing_data = []

    # Append new tools
    existing_data.extend(tools)

    # Write back
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

def Relevant_Tool_Generation(llm,target_tool_info,k,prompt,model,threshold,json_path):
    condition1 = 0
    condition2 = 0
    while not (condition1 and condition2):
        formatted_prompt = prompt.format(
                Description=target_tool_info,
                Integer = k
            )

        data = llm.generate(formatted_prompt)[0]

        tools = [(d["name"], d["description"]) for d in data]
        descriptions = [t["description"] for t in tools]
        target_desc = target_tool_info["description"]
        tool_embs = model.encode(descriptions, convert_to_tensor=True)
        target_emb = model.encode(target_desc, convert_to_tensor=True)

        cond1_list = []
        for emb in tool_embs:
            sim = cosine_similarity(emb, target_emb)
            cond1_list.append(sim >= threshold)

        condition1 = int(all(cond1_list))
        cond2_list = []
        num_tools = len(tool_embs)

        for i in range(num_tools):
            for j in range(i + 1, num_tools):
                sim = cosine_similarity(tool_embs[i], tool_embs[j])
                cond2_list.append(sim <= threshold + 0.1)

        condition2 = int(all(cond2_list))
        append_tools_to_json(tools, json_path)
