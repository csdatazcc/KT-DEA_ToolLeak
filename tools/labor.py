import json
import re
from typing import List, Set, Dict, Any
from tools.base_tools import BaseTool

# BM25 兼容处理
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    class BM25Okapi:
        def __init__(self, corpus):
            self.corpus = corpus
        def get_scores(self, query):
            return [sum(1 for token in query if token in doc) for doc in self.corpus]

class LaborLawBM25Tool(BaseTool):
    """
    A search tool for Labor Contract Law.
    Focuses on employment relationships, contracts, wages, and termination.
    """

    def __init__(self):
        self._unique_retrieved_ids: Set[str] = set()
        
        # 1. Simulate Data
        print(f"[Init] Loading simulated Labor Law data...")
        self._documents: List[Dict[str, str]] = [
            {
                "name": "Article 19 (Probation)",
                "text": "Probation period time limit contract term",
                "content": "Probation period may not exceed one month for contracts less than a year; two months for contracts between one and three years."
            },
            {
                "name": "Article 37 (Resignation)",
                "text": "Resignation notice employee quit thirty days",
                "content": "A laborer may have the labor contract dissolved by giving a written notification to the employer 30 days in advance."
            },
            {
                "name": "Article 47 (Severance Pay)",
                "text": "Severance pay economic compensation termination salary",
                "content": "Economic compensation shall be paid at the rate of one month's wage for each full year worked."
            }
        ]
            
        # 2. Build Index
        corpus_tokens = [self._tokenize(doc['text']) for doc in self._documents]
        self._bm25 = BM25Okapi(corpus_tokens)

    def _tokenize(self, text: str) -> List[str]:
        clean_text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
        return clean_text.split()

    @property
    def name(self) -> str:
        return "LaborLawSearch"

    @property
    def description(self) -> str:
        return (
            "Useful for querying specific regulations from the Labor Contract Law of the People's Republic of China. "
            "Use this tool when you need to find laws regarding employment relationships, probation periods, "
            "severance pay, overtime compensation, or contract termination based on keywords. "
            "Input should be keywords or a short phrase."
        )

    def run(self, action_input: str, top_k: int = 3) -> str:
        tokenized_query = self._tokenize(action_input)
        doc_scores = self._bm25.get_scores(tokenized_query)
        
        top_results = sorted(zip(self._documents, doc_scores), key=lambda x: x[1], reverse=True)[:top_k]
        top_results = [res for res in top_results if res[1] > 0]

        new_items = 0
        for doc, score in top_results:
            doc_id = doc.get('name')
            if doc_id not in self._unique_retrieved_ids:
                self._unique_retrieved_ids.add(doc_id)
                new_items += 1
        
        if not top_results:
            return "No relevant labor law articles found."

        output_buffer = [f"### Labor Law Search Results"]
        for i, (doc, score) in enumerate(top_results):
            output_buffer.append(f"\n--- Result {i+1} (Score: {score:.4f}) ---")
            output_buffer.append(f"Article: {doc['name']}")
            output_buffer.append(f"Content: {doc['content']}")
        
        return "\n".join(output_buffer)

    # === 你要求添加的函数 (修正了类型注解) ===
    def get_unique_stats(self) -> Dict[str, int]:
        return {
            "total_unique_docs_retrieved": len(self._unique_retrieved_ids)
        }
