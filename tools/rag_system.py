from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
from typing import Optional, List, Dict, Tuple, Callable
from tools.rag_database import RagDatabase,DPRagDatabase
from tools.utils import index_ints

class RAGRetriever:
    """
    完整 RAG 预处理流程：检索 →（可选）rerank → 格式化 → prompt 输出
    不调用大模型！只是把输入 LLM 的内容准备好返回给你
    """

    def __init__(
        self,
        database: RagDatabase,
        embedding_model: SentenceTransformer,
        reranker: Optional[FlagReranker] = None,
        format_rerank: Callable = lambda d: d["content"],
        format_retrieval: Callable = lambda d: d["content"],
        format_template: Callable = lambda docs, query, mode="default": [
            {"role": "system", "content": "你是一个知识检索助手，请根据文档回答问题"},
            {"role": "user", "content": f"参考文档:\n\n{''.join(docs)}\n\n问题: {query}"}
        ]
    ):
        self.database = database
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.format_rerank = format_rerank
        self.format_retrieval = format_retrieval
        self.format_template = format_template

    def fetch(
        self,
        query: str,
        n_retrieval: int = 16,
        n_rerank: int = 4,
        return_index: bool = False
    ) -> Tuple[List[str], List[float], Optional[list]]:
        """
        1) 向量检索 top-k
        2) reranker 精排 (optional)
        3) 返回精排后的 docs / scores / index
        """

        # Step 1: 向量检索
        retrieval, similarity, doc_idxs = self.database.retrieve_with_similarity(
            query, top_k=n_retrieval, return_index=True
        )

        # Step 2: Rerank （如果有）
        if self.reranker is not None:
            rerank_inputs = [(query, r) for r in self.format_rerank(retrieval)]
            scores = self.reranker.compute_score(rerank_inputs)
        else:
            scores = similarity[:n_rerank].tolist()

        # Step 3: 取前 n_rerank
        sorted_idx = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:n_rerank]
        final_indices = [i for i,_ in sorted_idx]
        final_scores = [s for _,s in sorted_idx]

        retrieval = {k: list(index_ints(v, final_indices)) for k,v in retrieval.items()}
        doc_idxs = index_ints(doc_idxs.tolist(), final_indices)

        docs = self.format_retrieval(retrieval)  # 抽 docs 列

        if return_index:
            return docs, final_scores, doc_idxs
        else:
            return docs, final_scores

    def prepare_prompt(
        self,
        query: str,
        n_retrieval: int = 16,
        n_rerank: int = 4,
        prompt_mode="default"
    ) -> Dict[str, object]:
        """
        主流程：返回可直接喂给大模型的 prompt + docs + scores
        """
        docs, scores, idxs = self.fetch(
            query,
            n_retrieval=n_retrieval,
            n_rerank=n_rerank,
            return_index=True
        )

        # 构造 chat template prompt
        chat_template = self.format_template(docs, query, prompt_mode)

        return {
            "query": query,
            "docs": docs,
            "scores": scores,
            "doc_indices": idxs,
            "prompt": chat_template
        }


class DPRAGRetriever:
    """
    与 RAGRetriever 用法完全一致，但底层数据库是 DP 的
    """

    def __init__(
        self,
        database: DPRagDatabase,
        embedding_model: SentenceTransformer,
        reranker: Optional[FlagReranker] = None,
        format_rerank: Callable = lambda d: d["content"],
        format_retrieval: Callable = lambda d: d["content"],
        format_template: Callable = lambda docs, query, mode="default": [
            {"role": "system", "content": "你是一个知识检索助手"},
            {"role": "user", "content": f"参考文档:\n\n{''.join(docs)}\n\n问题: {query}"}
        ]
    ):
        self.database = database
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.format_rerank = format_rerank
        self.format_retrieval = format_retrieval
        self.format_template = format_template

    def fetch(
        self,
        query: str,
        epsilon: float = 0.5,
        p: float = 0.1,
        alpha: float = 1.0,
        min_tau_bins: int = 100,
        n_rerank: int = 4,
        return_index: bool = True
    ) -> Tuple[List[str], List[float], Optional[list]]:

        retrieval, similarity, doc_idxs = self.database.retrieve_with_similarity(
            query,
            epsilon=epsilon,
            p=p,
            alpha=alpha,
            min_tau_bins=min_tau_bins,
            return_index=True
        )

        # rerank
        if self.reranker is not None:
            rerank_inputs = [(query, r) for r in self.format_rerank(retrieval)]
            scores = self.reranker.compute_score(rerank_inputs)
        else:
            scores = similarity.tolist()

        # top rerank
        sorted_idx = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:n_rerank]

        indices = [i for i, _ in sorted_idx]
        scores = [s for _, s in sorted_idx]

        retrieval = {k: list(index_ints(v, indices)) for k, v in retrieval.items()}
        doc_idxs = index_ints(doc_idxs.tolist(), indices)

        docs = self.format_retrieval(retrieval)

        return (docs, scores, doc_idxs) if return_index else (docs, scores)

    def prepare_prompt(
        self,
        query: str,
        epsilon: float = 0.5,
        p: float = 0.1,
        alpha: float = 1.0,
        min_tau_bins: int = 100,
        n_rerank: int = 4,
        prompt_mode="default"
    ) -> Dict[str, object]:

        docs, scores, idxs = self.fetch(
            query,
            epsilon, p, alpha, min_tau_bins,
            n_rerank=n_rerank,
            return_index=True
        )

        prompt = self.format_template(docs, query, prompt_mode)

        return {
            "query": query,
            "docs": docs,
            "scores": scores,
            "doc_indices": idxs,
            "prompt": prompt
        }