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

class BiomedicalLiteratureBM25Tool(BaseTool):
    """
    A search tool for retrieving broad biomedical academic literature/abstracts.
    Focuses on research papers rather than general advice.
    """

    def __init__(self):
        self._unique_retrieved_ids: Set[str] = set()
        
        # 1. Simulate Data (学术摘要)
        print(f"[Init] Loading simulated Biomedical Literature data...")
        self._documents: List[Dict[str, str]] = [
            {
                "name": "Study: Immunotherapy in Oncology (2023)",
                "text": "cancer immunotherapy pd-1 inhibitors immune checkpoint oncology treatment efficacy",
                "content": "Abstract: This study reviews the efficacy of PD-1 inhibitors in solid tumor treatment. Results indicate a 40% improvement in survival rates for specific subsets of patients with high mutational burden."
            },
            {
                "name": "Review: Gut Microbiome and Mental Health",
                "text": "microbiome gut brain axis depression anxiety probiotics bacteria intestine",
                "content": "Abstract: A systematic review of the gut-brain axis. Evidence suggests that gut microbiota composition significantly influences neurological signaling, potentially impacting depression and anxiety disorders."
            },
            {
                "name": "Trial: CRISPR gene editing for Sickle Cell",
                "text": "crispr gene editing cas9 sickle cell anemia genetic therapy blood",
                "content": "Abstract: Clinical trial results for CRISPR-Cas9 gene editing in patients with Sickle Cell Disease show sustained increases in fetal hemoglobin and elimination of vaso-occlusive crises."
            }
        ]
        
        # 2. Build Index
        corpus_tokens = [self._tokenize(doc['text']) for doc in self._documents]
        # 假设 BM25Okapi 已在上文定义或导入
        self._bm25 = BM25Okapi(corpus_tokens)

    def _tokenize(self, text: str) -> List[str]:
        clean_text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
        return clean_text.split()

    @property
    def name(self) -> str:
        return "BiomedicalLiteratureSearch"

    @property
    def description(self) -> str:
        return (
            "Useful for finding summaries of peer-reviewed research papers and academic studies. "
            "Input should be specific research topics, gene names, or therapy types (e.g., 'CRISPR', 'Immunotherapy'). "
            "Returns evidence from scientific literature rather than layman explanations."
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
            return "No relevant scientific literature found."

        output_buffer = [f"### Academic Literature Search Results"]
        for i, (doc, score) in enumerate(top_results):
            output_buffer.append(f"\n--- Result {i+1} (Score: {score:.4f}) ---")
            output_buffer.append(f"Source: {doc['name']}")
            output_buffer.append(f"Abstract: {doc['content']}")
        
        return "\n".join(output_buffer)

    def get_unique_stats(self) -> Dict[str, int]:
        return {
            "total_unique_papers_retrieved": len(self._unique_retrieved_ids)
        }