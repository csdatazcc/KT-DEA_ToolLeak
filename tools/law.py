import json
import re
from typing import List, Set, Dict, Any
from tools.base_tools import BaseTool
# 引入 rank_bm25 库
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("Please install rank_bm25 via 'pip install rank_bm25' to use this tool.")

class CivilCodeBM25Tool(BaseTool):
    """
    A search tool for the Civil Code of the People's Republic of China using BM25 algorithm.
    It performs keyword-based retrieval suitable for specific legal terms.
    """

    def __init__(self, json_path: str = "civil_code_200_bilingual.json"):
        """
        Initialize the BM25 Tool.

        :param json_path: Path to the .json file containing the legal articles.
        """
        self._json_path = json_path
        self._unique_retrieved_ids: Set[str] = set()
        
        # 1. Load Data
        print(f"[Init] Loading Civil Code data from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            self._documents: List[Dict[str, str]] = json.load(f)
            
        # 2. Preprocess and Tokenize for BM25
        # We search against the 'text' field (name + content)
        # Using a simple tokenizer (lowercasing and splitting by non-alphanumeric)
        corpus_tokens = [self._tokenize(doc['text']) for doc in self._documents]
        
        # 3. Build Index
        print(f"[Init] Building BM25 index for {len(self._documents)} articles...")
        self._bm25 = BM25Okapi(corpus_tokens)

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenizer helper: lowercase and split by whitespace/punctuation.
        """
        # Replace non-alphanumeric chars with space, then split
        clean_text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
        return clean_text.split()

    @property
    def name(self) -> str:
        return "ChineseCivilCodeSearch"

    @property
    def description(self) -> str:
        """
        Description for the Agent.
        """
        return (
            "Useful for querying specific articles from the Civil Code of the People's Republic of China. "
            "Use this tool when you need to find laws regarding contracts, marriage, guarantees, "
            "or property rights based on keywords. Input should be keywords or a short phrase."
        )

    def run(self, action_input: str, top_k: int = 3) -> str:
        """
        Executes the BM25 search.
        
        :param action_input: The search query (e.g., 'divorce cooling-off period').
        :param top_k: Number of documents to retrieve.
        """
        # 1. Tokenize Query
        tokenized_query = self._tokenize(action_input)
        
        # 2. Get Scores
        # We manually get scores to filter and sort
        doc_scores = self._bm25.get_scores(tokenized_query)
        
        # Zip scores with documents and sort by score descending
        top_results = sorted(
            zip(self._documents, doc_scores), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        # Filter out results with 0 score (irrelevant)
        top_results = [res for res in top_results if res[1] > 0]

        # 3. Update Unique Statistics
        new_items = 0
        for doc, score in top_results:
            # Use 'name' (Article ID) as the unique identifier
            doc_id = doc.get('name')
            if doc_id not in self._unique_retrieved_ids:
                self._unique_retrieved_ids.add(doc_id)
                new_items += 1
        
        total_unique = len(self._unique_retrieved_ids)

        # 4. Format Output
        if not top_results:
            return "No relevant articles found in the Civil Code for your query."

        output_buffer = [f"### Civil Code Search Results (Session Unique Records: {total_unique})"]
        
        output_buffer = []

        for i, (doc, score) in enumerate(top_results):
            output_buffer.append(f"\n--- Document {i+1} (BM25 Score: {score:.4f}) ---")
            output_buffer.append(f"Source: {doc['name']}")
            output_buffer.append(f"Content: {doc['content']}")
        
        output_buffer.append(f"\n[System]: {new_items} new unique articles accessed in this search.")
        #print(output_buffer)
        return "\n".join(output_buffer)

    def get_unique_stats(self) -> int:

        return {
            "total_unique_docs_retrieved": len(self._unique_retrieved_ids)
        }

# # ==========================================
# # Mock usage for demonstration
# # ==========================================
# if __name__ == "__main__":
#     # 2. Initialize Tool
#     tool = CivilCodeBM25Tool()

#     # 3. Run Query 1: Divorce (Marriage law)
#     print("\n>>> Query 1: 'divorce thirty days'")
#     print(tool.run("divorce thirty days"))

#     # 4. Run Query 2: Debt/Guarantee
#     print("\n>>> Query 2: 'guarantee principal claim'")
#     print(tool.run("guarantee principal claim"))
    
#     # 5. Run Query 3: Repeat logic to check unique count
#     print("\n>>> Query 3: 'divorce' (Should not increase unique count)")
#     print(tool.run("divorce"))