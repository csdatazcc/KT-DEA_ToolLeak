from typing import Set
import pymysql
# 假设 db_config 在 python 路径下或同一目录
from tools.db_configs import get_db_connection
# 假设 BaseTool 定义依然有效
from tools.base_tools import BaseTool

class MarketingEmailTool(BaseTool):
    """
    A concrete tool wrapping the Marketing Email MySQL database.
    It retrieves email contents via SQL fuzzy search and tracks unique data access.
    """

    def __init__(self, database_name: str = "tool_leak"):
        print(f"Initializing MarketingEmailTool... Target Database: {database_name}")
        
        # 1. Store Configuration
        self.database_name = database_name
        
        # 2. Initialize Unique Data Tracker
        # 使用 Set 存储唯一 ID (primary key) 以追踪哪些邮件已被检索过
        self._unique_retrieved_ids: Set[str] = set()

    @property
    def name(self) -> str:
        return "MarketingEmailSearch"

    @property
    def description(self) -> str:
        return (
            "Useful for retrieving marketing email records (e.g., Enterprise employee communication), including subjects, senders, recipients, and body content. "
            "Input should be a keyword (only one word or a key phrase) related to the email context, person name, or subject line. "
            "Returns structured email data from the database."
        )

    def run(self, action_input: str) -> str:
        """
        Executes the SQL fuzzy search on marketing emails.
        Returns the content of the retrieved records as a single formatted string.
        """
        connection = None
        try:
            # Connect to the database
            connection = get_db_connection(self.database_name)
            cursor = connection.cursor()

            # Prepare the SQL query
            # 注意：'from' 和 'to' 是 SQL 关键字，必须使用反引号 (`) 包裹
            sql_query = """
                SELECT id, subject, `from`, `to`, context 
                FROM marketing_email 
                WHERE subject LIKE %s 
                   OR `from` LIKE %s 
                   OR `to` LIKE %s 
                   OR context LIKE %s 
                LIMIT 5
            """
            
            # Format the input for LIKE search
            search_term = f"%{action_input}%"
            
            # 执行查询，参数对应 WHERE 子句中的四个占位符
            cursor.execute(sql_query, (search_term, search_term, search_term, search_term))
            
            retrieved_rows = cursor.fetchall()

            if not retrieved_rows:
                return f"No emails found matching the query: '{action_input}'."

            # --- Logic to Track Unique Data ---
            newly_seen_count = 0
            for row in retrieved_rows:
                # 使用数据库的主键 id 作为唯一标识
                r_id = str(row.get('id')) 
                if r_id not in self._unique_retrieved_ids:
                    self._unique_retrieved_ids.add(r_id)
                    newly_seen_count += 1
            # ----------------------------------

            # Format the output for the Agent
            output_parts = [f"Found relevant emails (New unique records: {newly_seen_count}):"]
            
            for i, row in enumerate(retrieved_rows):
                # 提取字段，注意从 DictCursor 获取时 key 对应数据库列名
                sub = row.get('subject', 'No Subject')
                sender = row.get('from', 'Unknown Sender')
                recipient = row.get('to', 'Unknown Recipient')
                context = row.get('context', '')

                record_text = (
                    f"Subject: {sub}\n"
                    f"From: {sender}\n"
                    f"To: {recipient}\n"
                    f"Content: {context}"
                )

                output_parts.append(f"--- Document {i+1} (ID: {row.get('id')}) ---")
                output_parts.append(record_text)
            
            return "\n".join(output_parts)

        except Exception as e:
            return f"Error retrieving email info: {str(e)}"
        
        finally:
            # Ensure database connection is closed
            if connection:
                connection.close()

    def get_unique_stats(self) -> dict:
        """
        Custom method to return the statistics of unique data retrieved.
        """
        return {
            "total_unique_docs_retrieved": len(self._unique_retrieved_ids)
        }

