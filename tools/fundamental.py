from typing import Set, List
import numpy as np
from sentence_transformers import SentenceTransformer, util
from tools.base_tools import BaseTool

class FundamentalAccountingTool(BaseTool):
    """
    A specific tool for retrieving quantitative financial metrics and 10-K data from the Fundamental RAG.
    It self-initializes the model and tracks unique data retrieval coverage.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print(f"[Init] Loading embedding model: {model_name}...")
        self._model = SentenceTransformer(model_name)
        
        # --- 内置基本面数据库 (Mocking RAG DB) ---
        # 这里的数据是具体的会计数字，区别于“新闻”
        self._knowledge_base = [
            {"title": "TSLA FY2023 Income Statement", "content": "Tesla reported total automotive revenues of $82.4 billion, with a gross margin of 18.2% for the fiscal year."},
            {"title": "AAPL Balance Sheet Q3", "content": "Apple's cash and cash equivalents stood at $166 billion. Total debt obligations were reduced by 5% to $105 billion."},
            {"title": "NVDA Earnings Metrics", "content": "Nvidia Data Center revenue grew 171% year-over-year to $14.5 billion. Diluted EPS was $4.02."},
            {"title": "AMZN Operating Cash Flow", "content": "Amazon operating cash flow increased 82% to $84.9 billion for the trailing twelve months."},
            {"title": "MSFT Cloud Segment", "content": "Microsoft Intelligent Cloud revenue was $24.3 billion, up 15% (up 17% in constant currency)."}
        ]
        
        self._corpus = [doc["content"] for doc in self._knowledge_base]
        self._corpus_embeddings = self._model.encode(self._corpus, convert_to_tensor=True)
        # ----------------------------------------
        
        self._unique_retrieved_docs: Set[str] = set()

    @property
    def name(self) -> str:
        return "fundamental_accounting_retriever"

    @property
    def description(self) -> str:
        """
        Description of the tool for the Agent.
        """
        # 语义高度相似，但关键词聚焦于“会计数据”、“指标”、“财报”
        return (
        "Useful for retrieving accounting metrics, financial statements, and earnings data "
        "from the fundamental analysis database. "
        "Input should be a search query or a question related to finance or economics."
    )

    def run(self, action_input: str) -> str:
        print(f"[Tool] Searching Fundamentals DB for: {action_input}")
        
        try:
            # 1. 向量检索
            query_embedding = self._model.encode(action_input, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, self._corpus_embeddings)[0]
            
            top_results = np.argpartition(-cos_scores.cpu(), range(min(3, len(self._corpus))))[0:3]
            top_results = sorted(top_results, key=lambda x: cos_scores[x], reverse=True)

            retrieved_docs = []
            titles = []
            scores = []

            for idx in top_results:
                score = float(cos_scores[idx])
                # 财务数据通常需要高精准度，阈值设高一点
                if score > 0.35: 
                    retrieved_docs.append(self._knowledge_base[idx]["content"])
                    titles.append(self._knowledge_base[idx]["title"])
                    scores.append(score)

            if not retrieved_docs:
                return "No relevant accounting data found."

            # 2. 记录唯一数据
            for doc in retrieved_docs:
                self._unique_retrieved_docs.add(doc)

            # 3. 格式化输出
            output_lines = ["Here is the relevant fundamental context retrieved:"]
            for i, doc in enumerate(retrieved_docs):
                output_lines.append(f"\n--- Result {i+1} (Score: {scores[i]:.4f}) ---")
                output_lines.append(f"Source: {titles[i]}")
                output_lines.append(f"Content: {doc}")

            return "\n".join(output_lines)

        except Exception as e:
            return f"Error executing fundamental retrieval: {str(e)}"

    def get_unique_retrieved_count(self) -> int:
        return len(self._unique_retrieved_docs)

    def get_coverage_report(self) -> str:
        count = len(self._unique_retrieved_docs)
        return f"Total unique financial records retrieved so far: {count}"