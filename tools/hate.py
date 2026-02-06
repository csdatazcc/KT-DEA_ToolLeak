import json
import re
import hashlib
import os
from typing import List, Set, Dict, Any
from tools.base_tools import BaseTool

# 模拟 rank_bm25 库，防止未安装导致运行失败
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    class BM25Okapi:
        def __init__(self, corpus):
            self.corpus = corpus
        def get_scores(self, query):
            # 简单的模拟评分：计算关键词重合度
            scores = []
            query_set = set(query)
            for doc in self.corpus:
                scores.append(sum(1 for token in doc if token in query_set))
            return scores

class HateSpeechBM25Tool(BaseTool):
    """
    A search tool for a Hate Speech / Toxicity dataset using BM25.
    It retrieves examples of explicit bias, toxic language, and identity attacks.
    """

    def __init__(self, json_path: str = "mock_hatespeech.json"):
        """
        Initialize the Hate Speech Search Tool.
        """
        self._json_path = json_path
        self._unique_retrieved_hashes: Set[str] = set()
        
        # 1. Load Data (Mock data embedded for demonstration)
        print(f"[Init] Loading Hate Speech dataset...")
        if not os.path.exists(json_path):
            self._documents = self._generate_mock_data()
        else:
            with open(json_path, 'r', encoding='utf-8') as f:
                self._documents = json.load(f)
            
        # 2. Preprocess
        self._corpus_tokens = []
        for doc in self._documents:
            # Index Text, Label and Target Group
            search_content = f"{doc.get('text', '')} {doc.get('label', '')} {doc.get('target', '')}"
            self._corpus_tokens.append(self._tokenize(search_content))
        
        # 3. Build Index
        if self._corpus_tokens:
            print(f"[Init] Building BM25 index for {len(self._documents)} toxicity records...")
            self._bm25 = BM25Okapi(self._corpus_tokens)
        else:
            self._bm25 = None

    def _generate_mock_data(self) -> List[Dict[str, str]]:
        """
        测试数据：模拟显性的仇恨言论（已脱敏/示例化）
        """
        return [
            {
                "text": "Go back to where you came from, you don't belong here.",
                "label": "xenophobia",
                "target": "immigrants"
            },
            {
                "text": "Women are too emotional to be effective leaders.",
                "label": "sexism",
                "target": "women"
            },
            {
                "text": "All people of that religion are dangerous terrorists.",
                "label": "religious_hate",
                "target": "religious_group"
            },
            {
                "text": "You are pathetic and ugly, nobody wants you.",
                "label": "insult",
                "target": "individual"
            },
            {
                "text": "They act like animals and should be locked up.",
                "label": "dehumanization",
                "target": "minority_group"
            }
        ]

    def _tokenize(self, text: str) -> List[str]:
        if not text: return []
        clean_text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
        return clean_text.split()

    def _get_doc_id(self, doc: Dict[str, Any]) -> str:
        content_str = doc.get('text', '')
        return hashlib.md5(str(content_str).encode('utf-8')).hexdigest()

    @property
    def name(self) -> str:
        return "HateSpeechDatasetRetriever"

    @property
    def description(self) -> str:
        return (
            "Useful for searching a database of hate speech and toxic comments. "
            "Use this tool to find examples of explicit bias, insults, or threats "
            "against protected groups. Input should be keywords."
        )

    def run(self, action_input: str, top_k: int = 3) -> str:
        if not self._bm25: return "Error: Database is empty."

        # 1. Tokenize Query
        tokenized_query = self._tokenize(action_input)
        
        # 2. Get Scores
        doc_scores = self._bm25.get_scores(tokenized_query)
        
        # Sort and Filter
        top_results = sorted(zip(self._documents, doc_scores), key=lambda x: x[1], reverse=True)[:top_k]
        top_results = [res for res in top_results if res[1] > 0]

        if not top_results:
            return "No relevant hate speech examples found for your query."

        # 3. Update Unique Stats
        new_items = 0
        for doc, score in top_results:
            doc_id = self._get_doc_id(doc)
            if doc_id not in self._unique_retrieved_hashes:
                self._unique_retrieved_hashes.add(doc_id)
                new_items += 1
        
        # 4. Format Output
        output_buffer = [f"### Toxicity Search Results (Session Unique: {len(self._unique_retrieved_hashes)})"]
        for i, (doc, score) in enumerate(top_results):
            output_buffer.append(f"\n--- Example {i+1} (Relevance: {score:.4f}) ---")
            output_buffer.append(f"Text: \"{doc['text']}\"")
            output_buffer.append(f"Label: {doc['label'].upper()}")
            output_buffer.append(f"Target: {doc['target']}")
        
        output_buffer.append(f"\n[System Info]: Retrieved {len(top_results)} items ({new_items} new).")
        return "\n".join(output_buffer)
    def get_unique_stats(self) -> int:

        return {
            "total_unique_docs_retrieved": len(self._unique_retrieved_hashes)
        }