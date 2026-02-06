from sentence_transformers import SentenceTransformer
from tools.rag_database import RagDatabase
from tools.rag_system import RAGRetriever
from typing import Set

# Assuming BaseTool is defined as provided in your second snippet
from tools.base_tools import BaseTool

class HealthcareRAGTool(BaseTool):
    """
    A concrete tool wrapping the HealthcareMagic RAG database.
    It retrieves relevant medical context and tracks unique data access.
    """

    def __init__(self, db_path: str = "rag_healthcaremagic_200.db", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print(f"Initializing HealthcareRAGTool... Loading model: {model_name}")
        
        # 1. Load Embedding Model
        self.embedding_model = SentenceTransformer(model_name)
        
        # 2. Load the RAG Database
        # Note: Ensure the db file exists at the path
        self.db = RagDatabase.load(db_path, self.embedding_model)
        
        # 3. Initialize Retriever
        self.rag = RAGRetriever(database=self.db, embedding_model=self.embedding_model)
        
        # 4. Initialize Unique Data Tracker
        # We use a Set to store unique document content strings (or IDs if available)
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
        """
        Executes the RAG retrieval.
        Returns the content of the retrieved documents as a single string.
        """
        try:
            # Using the method from your snippet
            # n_retrieval and n_rerank can be adjusted or made configurable
            result = self.rag.prepare_prompt(action_input, n_retrieval=5, n_rerank=3)
            
            retrieved_docs = result.get("docs", [])
            retrieved_scores = result.get("scores", [])
            
            if not retrieved_docs:
                return "No relevant medical documents found."

            # --- Logic to Track Unique Data ---
            newly_seen_count = 0
            for doc in retrieved_docs:
                # We use the document content string as the unique identifier
                if doc not in self._unique_retrieved_docs:
                    self._unique_retrieved_docs.add(doc)
                    newly_seen_count += 1
            # ----------------------------------

            # Format the output for the Agent
            output_parts = [f"Found relevant healthcare info (New unique records: {newly_seen_count}):"]
            
            for i, (doc, score) in enumerate(zip(retrieved_docs, retrieved_scores)):
                output_parts.append(f"--- Document {i+1} (Relevance: {score:.4f}) ---")
                output_parts.append(doc)
            
            return "\n".join(output_parts)

        except Exception as e:
            return f"Error retrieving medical info: {str(e)}"

    def get_unique_stats(self) -> dict:
        """
        Custom method to return the statistics of unique data retrieved.
        """
        return {
            "total_unique_docs_retrieved": len(self._unique_retrieved_docs)
        }