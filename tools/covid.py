from typing import Set
from sentence_transformers import SentenceTransformer
from tools.rag_database import RagDatabase
from tools.rag_system import RAGRetriever
from tools.base_tools import BaseTool
# 假设 BaseTool 定义在 base_tool.py 中
# from base_tool import BaseTool 

class CovidResearchTool(BaseTool):
    """
    A specific tool implementation for the TREC-COVID dataset.
    It retrieves scientific literature regarding COVID-19, SARS-CoV-2, and related coronaviruses.
    """

    def __init__(self, db_path: str = "rag_healthcaremagic_200.db", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the COVID-19 RAG tool.
        
        :param db_path: Path to the TREC-COVID vector database file.
        :param model_name: Name of the embedding model used for retrieval.
        """
        print(f"[Init] Loading embedding model: {model_name}...")
        self._model = SentenceTransformer(model_name)
        
        print(f"[Init] Loading TREC-COVID database from {db_path}...")
        self._db = RagDatabase.load(db_path, self._model)
        
        self._rag = RAGRetriever(database=self._db, embedding_model=self._model)
        
        # FEATURE: Set to store unique document contents retrieved across the session
        # 用于记录本次运行中所有被检索出来的唯一文档内容
        self._unique_retrieved_docs: Set[str] = set()

    @property
    def name(self) -> str:
        """
        The unique name of the tool. 
        Specific name helps the Agent distinguish it from general knowledge tools.
        """
        return "CovidScientificRetriever"

    @property
    def description(self) -> str:
        """
        Description for the Agent.
        Critically updated to focus on TREC-COVID content (COVID-19, SARS-CoV-2).
        """
        return (
            "Useful for answering questions about COVID-19, SARS-CoV-2, and related viruses. "
            "Input should be a specific question or keyword. Returns information from scientific studies, clinical trials, and research papers."
        )

    def run(self, action_input: str) -> str:
        """
        Executes the RAG retrieval logic.
        :param action_input: The query string regarding COVID-19.
        """
        # 1. Perform retrieval (using parameters suited for research papers)
        # n_retrieval=4, n_rerank=2 as per your example
        result = self._rag.prepare_prompt(action_input, n_retrieval=5, n_rerank=3)
        
        retrieved_docs = result["docs"]
        scores = result["scores"]
        
        # 2. Track unique documents (记录唯一数据)
        # We assume the document string content is the unique identifier.
        # If your DB has IDs, using IDs would be more memory efficient.
        new_items_count = 0
        for doc in retrieved_docs:
            if doc not in self._unique_retrieved_docs:
                self._unique_retrieved_docs.add(doc)
                new_items_count += 1
        
        total_unique_count = len(self._unique_retrieved_docs)
        
        # 3. Format the output for the English Agent
        output_buffer = [f"### Retrieval Result (Total unique records found in session: {total_unique_count})"]
        
        # Adding docs to output
        for i, (doc, score) in enumerate(zip(retrieved_docs, scores)):
            # Assuming the doc string contains the relevant text snippet
            output_buffer.append(f"--- Document {i+1} (Relevance Score: {score:.4f}) ---")
            output_buffer.append(f"Content: {doc}")
        
        output_buffer.append(f"\n[System Note]: {new_items_count} new unique documents added to session history.")
        
        # Join to form the final string response
        final_response = "\n".join(output_buffer)

        # Optional: Console log for debugging
        print(f"[Tool Log] Query: '{action_input}' | New Unique Docs: {new_items_count} | Total Unique: {total_unique_count}")

        return final_response

    def get_unique_doc_count(self) -> int:
        """
        External accessor to get the count of unique documents retrieved so far.
        """
        return len(self._unique_retrieved_docs)
    