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
            # 简单模拟：词频统计
            return [sum(1 for token in query if token in doc) for doc in self.corpus]

class CriminalCodeBM25Tool(BaseTool):
    """
    A search tool for the Criminal Law of the People's Republic of China.
    Focuses on crimes, penalties, and sentencing standards.
    """

    def __init__(self):
        self._unique_retrieved_ids: Set[str] = set()
        
        # 1. Simulate Data
        print(f"[Init] Loading simulated Criminal Law data...")
        self._documents: List[Dict[str, str]] = [
            {
                "name": "Article 264 (Theft)",
                "text": "Theft steal public private property money larceny",
                "content": "Whoever steals public or private property, if the amount is relatively large, shall be sentenced to fixed-term imprisonment of not more than three years."
            },
            {
                "name": "Article 263 (Robbery)",
                "text": "Robbery violence coercion force steal property",
                "content": "Whoever robs public or private property by violence, coercion or other methods shall be sentenced to fixed-term imprisonment."
            },
            {
                "name": "Article 232 (Intentional Homicide)",
                "text": "Intentional Homicide kill murder death penalty",
                "content": "Whoever intentionally kills another person shall be sentenced to death, life imprisonment or fixed-term imprisonment."
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
        return "CriminalLawSearch"

    @property
    def description(self) -> str:
        return (
            "Useful for querying specific articles from the Criminal Law of the People's Republic of China. "
            "Use this tool when you need to find definitions of crimes, sentencing standards, criminal liability, "
            "or penalties for offenses such as theft, robbery, homicide, or corruption based on keywords. "
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
            return "No relevant criminal articles found."

        output_buffer = [f"### Criminal Law Search Results"]
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
