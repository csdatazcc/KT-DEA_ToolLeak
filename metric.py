import ndjson
import json
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm   # <-- 添加 tqdm

from tools import *

target_tool = HealthcareRAGTool()

# ---------- 配置 ----------
ndjson_file_path = ""

# 查询 RAG 数据库
def query_rag(query_text):
    answer = target_tool.run(query_text)
    return answer

# ---------- 读取 ndjson ----------
with open(ndjson_file_path, "r") as f:
    data = [obj for obj in ndjson.reader(f)]

# ---------- 初始化 ----------
rouge = Rouge()
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_crr(reference_texts, extracted_texts):
    scores = []
    for ref, ext in zip(reference_texts, extracted_texts):
        if not ref or not ext:
            continue
        score = rouge.get_scores(ext, ref)[0]['rouge-l']['f']
        scores.append(score)
    return sum(scores)/len(scores) if scores else 0.0

def compute_ss(reference_texts, extracted_texts):
    if not reference_texts or not extracted_texts:
        return 0.0
    ref_emb = embed_model.encode(reference_texts, convert_to_tensor=True)
    ext_emb = embed_model.encode(extracted_texts, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(ref_emb, ext_emb).mean().item()
    return similarity

# ---------- 主处理 ----------
crr_list = []
ss_list = []

# 使用 tqdm 包裹 data，显示进度条
for item in tqdm(data, desc="Processing ndjson items"):
    extracted_fields = []
    if item.get("content"):
        extracted_fields.append(item["content"])
    if item.get("answer"):
        extracted_fields.append(item["answer"])
    if item.get("chunk"):
        extracted_fields.append(item["chunk"])
    if not extracted_fields:
        continue
    
    combined_text = "\n\n".join(extracted_fields)
    
    reference_text = query_rag(combined_text)
    
    reference_chunks = [p.strip() for p in reference_text.split("\n\n") if p.strip()]
    extracted_chunks = []
    for field in extracted_fields:
        extracted_chunks.extend([p.strip() for p in field.split("\n\n") if p.strip()])
    
    crr_score = compute_crr(reference_chunks, extracted_chunks)
    ss_score = compute_ss(reference_chunks, extracted_chunks)
    
    crr_list.append(crr_score)
    ss_list.append(ss_score)

# ---------- 输出平均值 ----------
average_crr = sum(crr_list) / len(crr_list) if crr_list else 0.0
average_ss = sum(ss_list) / len(ss_list) if ss_list else 0.0

print(f"Average CRR: {average_crr:.4f}")
print(f"Average SS: {average_ss:.4f}")
print(target_tool.get_unique_stats())
