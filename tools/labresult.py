import json
import re
from typing import List, Set, Dict, Any
from tools.base_tools import BaseTool

# --- BM25 兼容处理 (放在第一个工具中以确保环境可用) ---
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    class BM25Okapi:
        def __init__(self, corpus):
            self.corpus = corpus
        def get_scores(self, query):
            # 简单模拟：词频统计作为分数
            return [sum(1 for token in query if token in doc) for doc in self.corpus]


class LabResultInterpreterBM25Tool(BaseTool):
    """
    A search tool for interpreting medical lab test results.
    Focuses on reference ranges and clinical significance of metrics.
    """

    def __init__(self):
        self._unique_retrieved_ids: Set[str] = set()
        
        # 1. Simulate Data (检验指标 -> 含义)
        print(f"[Init] Loading simulated Lab Result data...")
        self._documents: List[Dict[str, str]] = [
            {
                "name": "ALT (Alanine Transaminase)",
                "text": "alt liver enzyme blood test hepatitis liver damage high",
                "content": "ALT is an enzyme found mostly in the liver. High levels indicate liver inflammation or damage (e.g., from hepatitis, alcohol, or fatty liver). Normal range is typically 7-56 units/liter."
            },
            {
                "name": "Hemoglobin (Hgb)",
                "text": "hemoglobin hgb blood anemia red blood cells oxygen low high iron",
                "content": "Hemoglobin is the protein in red blood cells that carries oxygen. Low levels indicate anemia (fatigue, weakness). High levels may occur in smokers or people living at high altitudes."
            },
            {
                "name": "TSH (Thyroid Stimulating Hormone)",
                "text": "tsh thyroid hormone hypothyroid hyperthyroid metabolism fatigue weight",
                "content": "TSH measures thyroid function. High TSH often indicates Hypothyroidism (underactive thyroid), causing fatigue and weight gain. Low TSH typically indicates Hyperthyroidism."
            },
            {
                "name": "Creatinine",
                "text": "creatinine kidney function renal blood waste urine kidneys",
                "content": "Creatinine is a waste product filtered by the kidneys. High levels in the blood suggest impaired kidney function or dehydration."
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
        return "LabResultInterpreter"

    @property
    def description(self) -> str:
        return (
            "Useful for interpreting medical laboratory test results. "
            "Input should be the name of the lab test (e.g., 'ALT', 'Hemoglobin', 'Creatinine') or related keywords. "
            "Returns the clinical meaning, normal patterns, and reasons for abnormal values."
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
            return "No relevant lab test information found."

        output_buffer = [f"### Lab Result Interpretation"]
        for i, (doc, score) in enumerate(top_results):
            output_buffer.append(f"\n--- Result {i+1} (Score: {score:.4f}) ---")
            output_buffer.append(f"Test Name: {doc['name']}")
            output_buffer.append(f"Interpretation: {doc['content']}")
        
        return "\n".join(output_buffer)

    def get_unique_stats(self) -> Dict[str, int]:
        return {
            "total_unique_lab_tests_retrieved": len(self._unique_retrieved_ids)
        }