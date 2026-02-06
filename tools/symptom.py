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

class SymptomAssessmentBM25Tool(BaseTool):
    """
    A search tool for mapping symptoms to potential medical conditions.
    Focuses on differential diagnosis based on patient complaints.
    """

    def __init__(self):
        self._unique_retrieved_ids: Set[str] = set()
        
        # 1. Simulate Data (症状 -> 可能的病症)
        print(f"[Init] Loading simulated Symptom Assessment data...")
        self._documents: List[Dict[str, str]] = [
            {
                "name": "Migraine (Headache)",
                "text": "headache throbbing pain sensitivity light nausea one side unilateral aura visual",
                "content": "Migraine: Often characterized by severe throbbing pain or a pulsing sensation, usually on just one side of the head. It's often accompanied by nausea, vomiting, and extreme sensitivity to light and sound."
            },
            {
                "name": "Gastroenteritis (Stomach Flu)",
                "text": "stomach pain diarrhea vomiting nausea fever abdominal cramps dehydration",
                "content": "Viral Gastroenteritis: An intestinal infection marked by watery diarrhea, abdominal cramps, nausea or vomiting, and sometimes fever. Commonly called the 'stomach flu'."
            },
            {
                "name": "Upper Respiratory Infection (Common Cold)",
                "text": "runny nose sore throat cough congestion mild fever sneezing fatigue",
                "content": "URI (Common Cold): Viral infection affecting the nose and throat. Symptoms are usually mild and resolve within a week. Key signs include runny nose, congestion, and sore throat."
            },
            {
                "name": "Myocardial Infarction (Heart Attack)",
                "text": "chest pain pressure shortness of breath arm pain jaw pain sweating anxiety",
                "content": "Myocardial Infarction: Requires immediate emergency care. Symptoms include chest pain/pressure, pain spreading to the arm or jaw, shortness of breath, and cold sweat."
            }
        ]
        
        # 2. Build Index
        corpus_tokens = [self._tokenize(doc['text']) for doc in self._documents]
        self._bm25 = BM25Okapi(corpus_tokens)

    def _tokenize(self, text: str) -> List[str]:
        # 简单的分词处理
        clean_text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
        return clean_text.split()

    @property
    def name(self) -> str:
        return "SymptomAssessmentSearch"

    @property
    def description(self) -> str:
        return (
            "Useful for analyzing reported physical symptoms to suggest potential causes. "
            "Input should be a list of symptoms or a description of the feeling (e.g., 'throbbing headache', 'stomach pain'). "
            "Returns potential conditions matching the symptom profile."
        )

    def run(self, action_input: str, top_k: int = 3) -> str:
        tokenized_query = self._tokenize(action_input)
        doc_scores = self._bm25.get_scores(tokenized_query)
        
        # 排序并取 Top K
        top_results = sorted(zip(self._documents, doc_scores), key=lambda x: x[1], reverse=True)[:top_k]
        # 过滤掉分数为 0 的结果
        top_results = [res for res in top_results if res[1] > 0]

        # 统计唯一检索条目
        new_items = 0
        for doc, score in top_results:
            doc_id = doc.get('name')
            if doc_id not in self._unique_retrieved_ids:
                self._unique_retrieved_ids.add(doc_id)
                new_items += 1
        
        if not top_results:
            return "No relevant conditions found for the described symptoms."

        output_buffer = [f"### Symptom Analysis Results"]
        for i, (doc, score) in enumerate(top_results):
            output_buffer.append(f"\n--- Result {i+1} (Score: {score:.4f}) ---")
            output_buffer.append(f"Potential Condition: {doc['name']}")
            output_buffer.append(f"Description: {doc['content']}")
        
        return "\n".join(output_buffer)

    def get_unique_stats(self) -> Dict[str, int]:
        return {
            "total_unique_conditions_retrieved": len(self._unique_retrieved_ids)
        }