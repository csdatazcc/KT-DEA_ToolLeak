from sentence_transformers import SentenceTransformer, util
import json
from Attack.key_word_v2 import ToolSemanticProcessor

PURPLE = "\033[35m"
CYAN = "\033[36m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_RED = "\033[91m"
WHITE = "\033[97m"
RESET = "\033[0m"


def expand_similar_tools(tool_datas, embeddings, start_index, threshold):
    queue = [start_index]
    visited = set([start_index])

    while queue:
        idx = queue.pop(0)
        for j in range(len(tool_datas)):
            if j not in visited:
                sim = float(util.cos_sim(embeddings[idx], embeddings[j]))
                if sim >= threshold:
                    visited.add(j)
                    queue.append(j)
    return visited

def Relevant_Tool_Selection(target_tool_name, tool_datas, threshold):
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model = SentenceTransformer(model_name)

    # Encode all descriptions
    descriptions = [t["description"] for t in tool_datas]
    embeddings = model.encode(descriptions, convert_to_tensor=True)

    # Find target index
    target_idx = next(i for i, t in enumerate(tool_datas) if t["name"] == target_tool_name)
    related_idx = expand_similar_tools(tool_datas, embeddings, target_idx, threshold)

    scores = []
    for i in related_idx:
        sim = float(util.cos_sim(embeddings[target_idx], embeddings[i]))
        scores.append((tool_datas[i]["name"], sim))

    relevant_tool = sorted(scores, key=lambda x: x[1], reverse=True)
    return relevant_tool
