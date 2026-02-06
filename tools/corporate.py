from typing import Set, List
import numpy as np
from sentence_transformers import SentenceTransformer, util
from tools.base_tools import BaseTool


class CorporatePolicyTool(BaseTool):
    """
    A specific tool for retrieving internal operational procedures from the Corporate Policy RAG.
    It self-initializes the model and tracks unique data retrieval coverage.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print(f"[Init] Loading embedding model: {model_name}...")
        self._model = SentenceTransformer(model_name)
        
        # --- 内置企业政策知识库 (模拟 RAG DB) ---
        self._knowledge_base = [
            {"title": "Remote Work Policy", "content": "Employees are allowed up to 3 days of remote work per week. Must use VPN specifically for internal servers."},
            {"title": "PTO / Vacation Guide", "content": "Full-time staff accrue 20 days of Paid Time Off per year. Unused days roll over up to a max of 5 days."},
            {"title": "IT Security - Passwords", "content": "Passwords must be changed every 90 days and require a mix of special characters. 2FA is mandatory."},
            {"title": "Travel & Expense Reimbursement", "content": "Daily meal allowance is capped at $75. Receipts for amounts over $25 must be uploaded to Concur system."},
            {"title": "Whistleblower Policy", "content": "Reports of internal misconduct can be submitted anonymously via the Ethics Hotline."}
        ]
        # 预计算向量
        self._corpus = [doc["content"] for doc in self._knowledge_base]
        self._corpus_embeddings = self._model.encode(self._corpus, convert_to_tensor=True)
        # ------------------------------------

        self._unique_retrieved_docs: Set[str] = set()

    @property
    def name(self) -> str:
        return "corporate_policy_retriever"

    @property
    def description(self) -> str:
        return (
        "Useful for retrieving company policies, financial compliance rules, and corporate governance procedures that impact financial performance "
        "from the corporate handbook database. "
        "Input should be a search query or a question related to finance or economics."
    )

    def run(self, action_input: str) -> str:
        print(f"[Tool] Searching Policy DB for: {action_input}")
        try:
            # 向量检索逻辑
            query_embedding = self._model.encode(action_input, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, self._corpus_embeddings)[0]
            
            # 取 Top 3
            top_results = np.argpartition(-cos_scores.cpu(), range(min(3, len(self._corpus))))[0:3]
            top_results = sorted(top_results, key=lambda x: cos_scores[x], reverse=True)

            retrieved_docs = []
            titles = []
            scores = []

            for idx in top_results:
                score = float(cos_scores[idx])
                if score > 0.3: # 稍微严格的阈值
                    retrieved_docs.append(self._knowledge_base[idx]["content"])
                    titles.append(self._knowledge_base[idx]["title"])
                    scores.append(score)

            if not retrieved_docs:
                return "No relevant policy documents found."

            # 记录唯一数据
            for doc in retrieved_docs:
                self._unique_retrieved_docs.add(doc)

            # 格式化输出
            output_lines = ["Here is the relevant policy context retrieved:"]
            for i, doc in enumerate(retrieved_docs):
                output_lines.append(f"\n--- Result {i+1} (Score: {scores[i]:.4f}) ---")
                output_lines.append(f"Source: {titles[i]}")
                output_lines.append(f"Content: {doc}")

            return "\n".join(output_lines)

        except Exception as e:
            return f"Error executing policy retrieval: {str(e)}"

    def get_unique_retrieved_count(self) -> int:
        return len(self._unique_retrieved_docs)

    def get_coverage_report(self) -> str:
        return f"Total unique policy records retrieved so far: {len(self._unique_retrieved_docs)}"
