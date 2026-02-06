import os, json, tqdm
import torch
from typing import Dict, List, Union, Optional, Tuple
from sentence_transformers import SentenceTransformer

class RagDatabase:
    """
    简单可用的 RAG 向量数据库:
    - 文本 → embedding → 存储
    - TopK 语义检索
    - 可保存 & 加载
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        primary_key_embeddings: torch.Tensor,
        columns: Dict[str, List]
    ):
        self.embedding_model = embedding_model
        self.primary_key_embeddings = primary_key_embeddings  # [N, d]
        self.columns = columns  # {"content":[...], "title":[...]...}

    # ========= 核心检索 =========
    def retrieve_index_and_similarity(
        self, query: Union[str, torch.Tensor], top_k:int=4
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if isinstance(query, str):
            query = self.embedding_model.encode(query, convert_to_tensor=True)

        similarity = torch.linalg.vecdot(query, self.primary_key_embeddings)  # dot sim
        scores, idxs = torch.topk(similarity, top_k)
        return idxs, scores

    def retrieve_with_similarity(
        self, query: Union[str, torch.Tensor], top_k:int=4, return_index=False
    ):
        idxs, scores = self.retrieve_index_and_similarity(query, top_k)
        result = {k: [v[i] for i in idxs] for k, v in self.columns.items()}
        print(result)
        return (result, scores, idxs) if return_index else (result, scores)

    def retrieve(self, query, top_k=4):
        docs, _ = self.retrieve_with_similarity(query, top_k)
        return docs

    # ========= 构建数据库 =========
    @classmethod
    def from_texts(
        cls, 
        embedding_model: SentenceTransformer, 
        texts: List[str], 
        extra_columns: Optional[Dict[str, List]] = None, 
        batch_size: int = 16
    ):
        """
        构建向量库
        `texts` 作为主文本，对应 content
        """
        n = len(texts)
        columns = {"content": texts}

        # 如果补充列（如标题/ID）
        if extra_columns:
            for k,v in extra_columns.items():
                assert len(v) == n
                columns[k] = v

        # 计算 embeddings
        embs = []
        for i in tqdm.tqdm(range(0, n, batch_size), desc="Embedding"):
            batch = texts[i:i+batch_size]
            embs.append(
                embedding_model.encode(batch, convert_to_tensor=True, normalize_embeddings=True)
            )
        embs = torch.cat(embs, dim=0)

        return cls(embedding_model, embs, columns)

    # ========= 保存 & 加载 =========
    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.primary_key_embeddings, os.path.join(save_dir, "primary_keys.pth"))
        with open(os.path.join(save_dir, "columns.json"), "w") as f:
            json.dump(self.columns, f, ensure_ascii=False)

    @classmethod
    def load(cls, load_dir, embedding_model):
        device = embedding_model.device
        pk = torch.load(os.path.join(load_dir, "primary_keys.pth")).to(device)
        columns = json.load(open(os.path.join(load_dir, "columns.json")))
        return cls(embedding_model, pk, columns)


class DPRagDatabase:
    """
    与 RagDatabase 接口对齐的 DP 版本
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        primary_key_embeddings: torch.Tensor,
        columns: Dict[str, List]
    ):
        self.embedding_model = embedding_model
        self.primary_key_embeddings = primary_key_embeddings  # [N, d]
        self.columns = columns

    def dp_retrieve_index_and_similarity(
        self,
        query: Union[str, torch.Tensor],
        epsilon: float,
        p: float = 0.1,
        alpha: float = 1.0,
        min_tau_bins: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基于 Exponential Mechanism 的 DP Retrieval
        """

        # 1. 编码 query
        if isinstance(query, str):
            query = self.embedding_model.encode(
                query, convert_to_tensor=True, normalize_embeddings=True
            )

        # 2. 相似度
        similarity = torch.matmul(self.primary_key_embeddings, query)

        # 3. 排序
        sorted_scores, sorted_indices = torch.sort(similarity, descending=True)

        # 4. 权重
        s_max, s_min = sorted_scores.max(), sorted_scores.min()
        delta = (s_max - s_min).clamp(min=1e-6)
        weights = torch.exp(alpha * (sorted_scores - s_max) / delta)

        # 5. tau 离散化
        tau_candidates = torch.linspace(
            s_min.item(), s_max.item(), min_tau_bins,
            device=sorted_scores.device
        )

        # 6. utility 计算
        utilities = []
        total_weight = weights.sum()
        for tau in tau_candidates:
            mask = sorted_scores >= tau
            selected_weight = weights[mask].sum()
            utility = -torch.abs(selected_weight - p * total_weight)
            utilities.append(utility)
        utilities = torch.stack(utilities)

        # 7. Exponential mechanism
        probs = torch.exp(epsilon * utilities / 2)
        probs = probs / probs.sum()

        tau = tau_candidates[torch.multinomial(probs, 1)]
        mask = sorted_scores >= tau

        return sorted_indices[mask], sorted_scores[mask]

    def retrieve_with_similarity(
        self,
        query: Union[str, torch.Tensor],
        epsilon: float,
        p: float = 0.1,
        alpha: float = 1.0,
        min_tau_bins: int = 100,
        return_index: bool = False
    ):
        idxs, scores = self.dp_retrieve_index_and_similarity(
            query, epsilon, p, alpha, min_tau_bins
        )

        result = {k: [v[i] for i in idxs] for k, v in self.columns.items()}
        return (result, scores, idxs) if return_index else (result, scores)
    
    @classmethod
    def load(cls, load_dir: str, embedding_model):
        """
        与 RagDatabase.load 完全一致
        只是返回 DPRagDatabase 实例
        """
        device = embedding_model.device

        primary_key_embeddings = torch.load(
            os.path.join(load_dir, "primary_keys.pth"),
            map_location=device
        )

        with open(os.path.join(load_dir, "columns.json"), "r", encoding="utf-8") as f:
            columns = json.load(f)

        return cls(
            embedding_model=embedding_model,
            primary_key_embeddings=primary_key_embeddings,
            columns=columns
        )