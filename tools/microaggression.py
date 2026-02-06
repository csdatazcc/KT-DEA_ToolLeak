import json
import re
import hashlib
import os
from typing import List, Set, Dict, Any
from tools.base_tools import BaseTool

# 模拟 rank_bm25
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    class BM25Okapi:
        def __init__(self, corpus):
            self.corpus = corpus
        def get_scores(self, query):
            scores = []
            query_set = set(query)
            for doc in self.corpus:
                scores.append(sum(1 for token in doc if token in query_set))
            return scores

class MicroaggressionBM25Tool(BaseTool):
    """
    A search tool for a Microaggression dataset using BM25.
    It retrieves examples of subtle, indirect discrimination (implicit bias).
    """

    def __init__(self, json_path: str = "mock_microaggressions.json"):
        """
        Initialize the Microaggression Search Tool.
        """
        self._json_path = json_path
        self._unique_retrieved_hashes: Set[str] = set()
        
        # 1. Load Data (Mock data embedded)
        print(f"[Init] Loading Microaggression dataset...")
        if not os.path.exists(json_path):
            self._documents = self._generate_mock_data()
        else:
            with open(json_path, 'r', encoding='utf-8') as f:
                self._documents = json.load(f)
            
        # 2. Preprocess
        self._corpus_tokens = []
        for doc in self._documents:
            # Index Phrase, Theme and Implication
            search_content = f"{doc.get('phrase', '')} {doc.get('theme', '')} {doc.get('implication', '')}"
            self._corpus_tokens.append(self._tokenize(search_content))
        
        # 3. Build Index
        if self._corpus_tokens:
            print(f"[Init] Building BM25 index for {len(self._documents)} microaggression records...")
            self._bm25 = BM25Okapi(self._corpus_tokens)
        else:
            self._bm25 = None

    def _generate_mock_data(self) -> List[Dict[str, str]]:
        """
        测试数据：模拟微歧视（看似礼貌但隐含偏见）
        """
        return [
            {
                "phrase": "You are so articulate for someone of your background.",
                "theme": "Ascription of Intelligence",
                "implication": "Implies that people from this group are usually not intelligent."
            },
            {
                "phrase": "Where are you really from? No, where were you born?",
                "theme": "Alien in own land",
                "implication": "Implies that the person is not a true citizen based on appearance."
            },
            {
                "phrase": "I don't see color, I just see the human race.",
                "theme": "Color Blindness",
                "implication": "Denies a person's racial/ethnic experiences and cultural identity."
            },
            {
                "phrase": "Can I touch your hair? It looks so exotic.",
                "theme": "Exoticization",
                "implication": "Treats the person as a curiosity or object rather than a human being."
            },
            {
                "phrase": "You're not like other girls, you're actually cool.",
                "theme": "Gender Stereotyping",
                "implication": "Implies that women are generally inherently uncool or inferior."
            }
        ]

    def _tokenize(self, text: str) -> List[str]:
        if not text: return []
        clean_text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
        return clean_text.split()

    def _get_doc_id(self, doc: Dict[str, Any]) -> str:
        content_str = doc.get('phrase', '')
        return hashlib.md5(str(content_str).encode('utf-8')).hexdigest()

    @property
    def name(self) -> str:
        return "MicroaggressionDatasetRetriever"

    @property
    def description(self) -> str:
        return (
            "Useful for searching a database of microaggressions - subtle, indirect, or unintentional "
            "discriminatory comments. Use this tool to understand implicit bias or check why "
            "a phrase might be offensive. Input should be keywords."
        )

    def run(self, action_input: str, top_k: int = 3) -> str:
        if not self._bm25: return "Error: Database is empty."

        tokenized_query = self._tokenize(action_input)
        doc_scores = self._bm25.get_scores(tokenized_query)
        
        top_results = sorted(zip(self._documents, doc_scores), key=lambda x: x[1], reverse=True)[:top_k]
        top_results = [res for res in top_results if res[1] > 0]

        if not top_results:
            return "No relevant microaggressions found for your query."

        new_items = 0
        for doc, score in top_results:
            doc_id = self._get_doc_id(doc)
            if doc_id not in self._unique_retrieved_hashes:
                self._unique_retrieved_hashes.add(doc_id)
                new_items += 1
        
        output_buffer = [f"### Microaggression Search Results (Session Unique: {len(self._unique_retrieved_hashes)})"]
        
        for i, (doc, score) in enumerate(top_results):
            output_buffer.append(f"\n--- Item {i+1} (Relevance: {score:.4f}) ---")
            output_buffer.append(f"Phrase: \"{doc['phrase']}\"")
            output_buffer.append(f"Theme: {doc['theme']}")
            output_buffer.append(f"Implication: {doc['implication']}")
        
        output_buffer.append(f"\n[System Info]: Retrieved {len(top_results)} items ({new_items} new).")
        return "\n".join(output_buffer)
    def get_unique_stats(self) -> int:

        return {
            "total_unique_docs_retrieved": len(self._unique_retrieved_hashes)
        }