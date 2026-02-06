# main.py
from drug import DrugReferenceTool
from clinical import ClinicalGuidelineTool

def mock_agent_execution():
    print(">>> INITIALIZING MEDICAL AGENT TOOLS...")
    # 1. 实例化工具
    drug_tool = DrugReferenceTool()
    guideline_tool = ClinicalGuidelineTool()
    
    # 模拟 Agent 的用户输入队列
    user_queries = [
        # --- 场景 1: 具体的药物查询 (包含别名/品牌名) ---
        ("drug", "Zofran side effects"),           # 应匹配 Ondansetron
        ("drug", "blood thinner Eliquis"),         # 应匹配 Apixaban
        ("drug", "pain killer gabapentin"),        # 应匹配 Gabapentin
        
        # --- 场景 2: 模糊查询 (拼写错误/非正式名称) ---
        ("drug", "metformin"),                     # 拼写错误，仍应匹配 Metformin
        ("drug", "adrenalin dose"),                # 品牌名，应匹配 Epinephrine
        
        # --- 场景 3: 疾病指南查询 (缩写/关键词) ---
        ("guide", "management of COPD"),           # 应匹配 COPD (GOLD)
        ("guide", "heart failure treatment"),      # 应匹配 Heart Failure (HFrEF)
        ("guide", "stroke protocol"),              # 应匹配 Ischemic Stroke
        
        # --- 场景 4: 正交性测试 (同一个词，不同工具) ---
        # 用户问 "Pneumonia" 的时候，查药只能查到抗生素，查指南能查到流程
        ("drug", "Pneumonia treatment"),           # 可能会匹配到 Amoxicillin (因为Indication里有Pneumonia) 或无结果
        ("guide", "Pneumonia treatment")           # 肯定匹配到 CAP Guidelines
    ]

    print("\n>>> STARTING BATCH PROCESSING...\n")

    for intent, query in user_queries:
        print(f"USER ASKED: '{query}'")
        
        if intent == "drug":
            print(f"AGENT ACTION: Calling {drug_tool.name}...")
            result = drug_tool.run(query)
        else:
            print(f"AGENT ACTION: Calling {guideline_tool.name}...")
            result = guideline_tool.run(query)
            
        # 打印简略结果 (只取前3行用于展示，避免刷屏)
        preview = "\n".join(result.split("\n")[:5]) 
        print(f"TOOL OUTPUT:\n{preview}...\n")
        print("-" * 60)

if __name__ == "__main__":
    mock_agent_execution()