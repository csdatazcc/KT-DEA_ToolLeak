from typing import Set, List, Optional
from sentence_transformers import SentenceTransformer
from tools.rag_database import RagDatabase
from tools.rag_system import RAGRetriever
from tools.base_tools import BaseTool

class FinancialKnowledgeTool(BaseTool):
    """
    A specific tool for retrieving financial information from the Financial Phrasebank RAG.
    It self-initializes the model and database, and tracks unique data retrieval coverage.
    """

    def __init__(self, db_path: str = "rag_financial_phrasebank_200.db", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the Financial RAG tool.
        
        :param db_path: Path to the financial vector database file.
        :param model_name: Name of the embedding model used for retrieval.
        """
        print(f"Initializing FinancialRAGTool... Loading model: {model_name}")
        
        # 1. Load Embedding Model
        self.embedding_model = SentenceTransformer(model_name)
        
        # 2. Load the RAG Database
        self.db = RagDatabase.load(db_path, self.embedding_model)
        
        # 3. Initialize Retriever
        self.rag = RAGRetriever(database=self.db, embedding_model=self.embedding_model)
        
        # 4. Initialize Unique Data Tracker
        self._unique_retrieved_docs: Set[str] = set()
    @property
    def name(self) -> str:
        """The unique name of the tool."""
        return "financial_phrasebank_retriever"

    @property
    def description(self) -> str:
        """
        Description of the tool for the Agent.
        """
        return (
            "Useful for retrieving financial news, sentiment analysis, and market context "
            "from the financial phrasebank database. "
            "Input should be a search query or a question related to finance or economics."
        )

    def run(self, action_input: str) -> str:
        """
        Executes the RAG retrieval.
        Returns the content of the retrieved documents as a single string.
        """
        try:
            # Retrieve documents (adjust n_retrieval and n_rerank as needed)
            result = self.rag.prepare_prompt(action_input, n_retrieval=5, n_rerank=1)
            
            retrieved_docs = result.get("docs", [])
            retrieved_scores = result.get("scores", [])

            if not retrieved_docs:
                return "No relevant financial documents found."

            # --- Track Unique Documents ---
            newly_seen_count = 0
            for doc in retrieved_docs:
                if doc not in self._unique_retrieved_docs:
                    self._unique_retrieved_docs.add(doc)
                    newly_seen_count += 1
            # --------------------------------

            # Format output for the Agent
            output_parts = [f"Found relevant financial info (New unique records: {newly_seen_count}):"]
            
            for i, (doc, score) in enumerate(zip(retrieved_docs, retrieved_scores)):
                output_parts.append(f"--- Document {i+1} (Relevance: {score:.4f}) ---")
                output_parts.append(doc)
            
            return "\n".join(output_parts)

        except Exception as e:
            return f"Error retrieving financial info: {str(e)}"

    def get_unique_stats(self) -> dict:
        """
        Returns statistics about unique documents retrieved.
        """

        return {
            "total_unique_docs_retrieved": len(self._unique_retrieved_docs)
        }