from sentence_transformers import SentenceTransformer
from tools.rag_database import DPRagDatabase
from tools.rag_system import DPRAGRetriever
from typing import Set
from tools.base_tools import BaseTool

class HealthcareRAGToolDP(BaseTool):
    """
    DP-RAG version of Healthcare tool
    """

    def __init__(
        self,
        db_path: str = "rag_healthcaremagic_200.dpdb",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        epsilon: float = 0.5
    ):
        print(f"Initializing HealthcareDPRAGTool (ε={epsilon})")

        self.embedding_model = SentenceTransformer(model_name)

        # === Load DP database ===
        self.db = DPRagDatabase.load(db_path, self.embedding_model)

        # === DP Retriever ===
        self.rag = DPRAGRetriever(
            database=self.db,
            embedding_model=self.embedding_model
        )

        self.epsilon = epsilon

        # === Unique data tracker ===
        self._unique_retrieved_docs: Set[str] = set()

    @property
    def name(self) -> str:
        return "HealthcareKnowledgeSearch"

    @property
    def description(self) -> str:
        return (
            "Useful for answering questions about medical conditions, symptoms, treatments, and general healthcare advice. "
            "Input should be a specific medical question or keyword query. Returns concise, evidence-based explanations."
        )

    def run(self, action_input: str) -> str:
        try:
            docs, scores, _ = self.rag.fetch(
                query=action_input,
                epsilon=self.epsilon,
                n_rerank=3
            )

            if not docs:
                return "No relevant medical documents found."

            newly_seen = 0
            for d in docs:
                if d not in self._unique_retrieved_docs:
                    self._unique_retrieved_docs.add(d)
                    newly_seen += 1

            output = [
                f"Found DP-protected healthcare info (ε={self.epsilon}, New records={newly_seen})"
            ]

            for i, (doc, score) in enumerate(zip(docs, scores)):
                output.append(f"--- Document {i+1} (Score={score:.4f}) ---")
                output.append(doc)

            return "\n".join(output)

        except Exception as e:
            return f"DP-RAG error: {str(e)}"

    def get_unique_stats(self):
        return {"total_unique_docs_retrieved": len(self._unique_retrieved_docs)}
