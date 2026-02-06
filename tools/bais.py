import json
import re
import hashlib
from typing import List, Set, Dict, Any, Optional

# 假设 base_tools 已经存在，如果是一个独立文件运行，需要取消下面 BaseTool 的注释并移除 import
from tools.base_tools import BaseTool
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("Please install rank_bm25 via 'pip install rank_bm25' to use this tool.")

class GrepBiasBM25Tool(BaseTool):
    """
    A search tool for the GrepBias dataset (or similar bias corpora) using BM25.
    It retrieves examples of social biases (gender, race, religion, etc.) based on keywords.
    """

    def __init__(self, json_path: str = "GrepBias_200.json"):
        """
        Initialize the Bias Search Tool.
        
        :param json_path: Path to the .json file containing the bias dataset.
        """
        self._json_path = json_path
        
        # Store hashes of retrieved docs to track uniqueness
        # 用于记录历史唯一数据，但不影响单词查询的返回结果
        self._unique_retrieved_hashes: Set[str] = set()
        
        # 1. Load Data
        print(f"[Init] Loading Bias dataset from {json_path}...")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self._documents: List[Dict[str, Any]] = json.load(f)
        except FileNotFoundError:
            print(f"[Warning] File {json_path} not found. Please ensure the file exists.")
            self._documents = []
            
        # 2. Preprocess and Tokenize
        # The provided JSON uses 'Text' as the combined field suitable for indexing.
        self._corpus_tokens = []
        for doc in self._documents:
            # Use 'Text' for the search index as it contains both Title and Document
            search_content = doc.get('Text', doc.get('Document', '')) 
            self._corpus_tokens.append(self._tokenize(search_content))
        
        # 3. Build Index
        if self._corpus_tokens:
            print(f"[Init] Building BM25 index for {len(self._documents)} records...")
            self._bm25 = BM25Okapi(self._corpus_tokens)
        else:
            self._bm25 = None

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenizer: lowercase and split by non-alphanumeric characters.
        """
        if not text:
            return []
        clean_text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
        return clean_text.split()

    def _get_doc_id(self, doc: Dict[str, Any]) -> str:
        """
        Helper to identify a unique document using MD5 hash of its Text.
        """
        content_str = doc.get('Text', doc.get('Document', ''))
        return hashlib.md5(str(content_str).encode('utf-8')).hexdigest()

    @property
    def name(self) -> str:
        return "SocialBiasDatasetRetriever"

    @property
    def description(self) -> str:
        return (
            "Useful for searching a database of social bias examples, stereotypes, "
            "and prejudicial text. Use this tool when you need to find specific examples "
            "of gender, racial, or religious bias. Input should be keywords."
        )

    def run(self, action_input: str, top_k: int = 3) -> str:
        """
        Executes the BM25 search logic.
        WARNING: Will return top_k results regardless of whether they have been seen before.
        """
        if not self._bm25:
            return "Error: Database is empty or not initialized."

        # 1. Tokenize Query
        tokenized_query = self._tokenize(action_input)
        
        # 2. Get Scores
        doc_scores = self._bm25.get_scores(tokenized_query)
        
        # Sort by score descending
        top_results = sorted(
            zip(self._documents, doc_scores), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        # Filter zero-score results (irrelevant)
        top_results = [res for res in top_results if res[1] > 0]

        if not top_results:
            return "No relevant bias examples found for your query."

        # 3. Update Unique Statistics (Background Tracking)
        # 注意：这里只更新计数，不影响 top_results 的内容
        new_items = 0
        for doc, score in top_results:
            doc_id = self._get_doc_id(doc)
            if doc_id not in self._unique_retrieved_hashes:
                self._unique_retrieved_hashes.add(doc_id)
                new_items += 1
        
        total_unique = len(self._unique_retrieved_hashes)

        # 4. Format Output (Always outputs all top_results)
        output_buffer = [f"### Bias Dataset Search Results (Session Unique Records: {total_unique})"]
        output_buffer = []
        for i, (doc, score) in enumerate(top_results):
            output_buffer.append(f"\n--- Source {i+1} (Relevance: {score:.4f}) ---")
            
            if 'Title' in doc and doc['Title']:
                output_buffer.append(f"Title: {doc['Title']}")
            
            if 'Document' in doc:
                output_buffer.append(f"Document: {doc['Document']}")
            elif 'Text' in doc:
                output_buffer.append(f"Text: {doc['Text']}")
        
        # 系统提示信息，告知有多少是新发现的，但不影响主要内容的展示
        output_buffer.append(f"\n[System Info]: Retrieved {len(top_results)} items ({new_items} new to this session).")
        
        return "\n".join(output_buffer)
    def get_unique_stats(self) -> int:

        return {
            "total_unique_docs_retrieved": len(self._unique_retrieved_hashes)
        }