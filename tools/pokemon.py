from typing import Set, List, Dict, Any
import pymysql
# Assuming db_config is in the python path or same directory
from tools.db_configs import get_db_connection
# Assuming BaseTool is defined as per your context
from tools.base_tools import BaseTool

class PokemonDatabaseTool(BaseTool):
    """
    A concrete tool wrapping the Pokemon MySQL database.
    It retrieves Pokemon details via SQL fuzzy search and tracks unique data access.
    """

    def __init__(self, database_name: str = "tool_leak"):
        print(f"Initializing PokemonDatabaseTool... Target Database: {database_name}")
        
        # 1. Store Configuration
        self.database_name = database_name
        
        # 2. Initialize Unique Data Tracker
        # We use a Set to store unique Pokemon IDs (primary key) to track what has been seen.
        self._unique_retrieved_ids: Set[str] = set()

    @property
    def name(self) -> str:
        return "PokemonInfoSearch"

    @property
    def description(self) -> str:
        return (
            "Useful for retrieving details about Pokemon including their type, physical description, and capabilities. "
            "Input should be a Pokemon name, type (e.g., 'fire', 'grass'), or one keyword description. "
            "Returns structured information from the database."
        )

    def run(self, action_input: str) -> str:
        """
        Executes the SQL fuzzy search.
        Returns the content of the retrieved records as a single formatted string.
        """
        connection = None
        try:
            # Connect to the database
            connection = get_db_connection(self.database_name)
            cursor = connection.cursor()

            # Prepare the SQL query for fuzzy matching across specified columns
            # We limit the result to 5 as requested
            sql_query = """
                SELECT id, name, type_1, type_2, caption 
                FROM pokemon_data 
                WHERE name LIKE %s 
                   OR type_1 LIKE %s 
                   OR type_2 LIKE %s 
                   OR caption LIKE %s 
                LIMIT 5
            """
            
            # Format the input for LIKE search (e.g., '%pikachu%')
            search_term = f"%{action_input}%"
            cursor.execute(sql_query, (search_term, search_term, search_term, search_term))
            
            retrieved_rows = cursor.fetchall()

            if not retrieved_rows:
                return f"No Pokemon found matching the query: '{action_input}'."

            # --- Logic to Track Unique Data ---
            newly_seen_count = 0
            for row in retrieved_rows:
                # We use the unique 'id' column from the database as the identifier
                p_id = str(row.get('id')) 
                if p_id not in self._unique_retrieved_ids:
                    self._unique_retrieved_ids.add(p_id)
                    newly_seen_count += 1
            # ----------------------------------

            # Format the output for the Agent
            output_parts = [f"Found relevant Pokemon info (New unique records: {newly_seen_count}):"]
            
            for i, row in enumerate(retrieved_rows):
                # Constructing a readable format for the LLM
                p_name = row.get('name', 'Unknown')
                p_type1 = row.get('type_1', '')
                p_type2 = row.get('type_2', '')
                p_caption = row.get('caption', '')
                
                # Handling type formatting
                types = p_type1
                if p_type2 and p_type2.lower() != 'none' and p_type2.strip() != '':
                    types += f" / {p_type2}"

                record_text = (
                    f"Name: {p_name}\n"
                    f"Type: {types}\n"
                    f"Description: {p_caption}"
                )

                output_parts.append(f"--- Document {i+1} (ID: {row.get('id')}) ---")
                output_parts.append(record_text)
            
            return "\n".join(output_parts)

        except Exception as e:
            return f"Error retrieving Pokemon info: {str(e)}"
        
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

# ==========================================
# Usage Example
# ==========================================
# if __name__ == "__main__":
#     # 1. Initialize the tool
#     # 请确保数据库 'pokemon_db' 存在，并且其中包含 pokemon_data 表
#     # 如果你的数据库名字不一样，请修改 database_name 参数
#     tool = PokemonDatabaseTool()

#     # 2. Simulate Agent calls
#     print("\n>>> Agent Query 1: Search for 'Grass' type or description")
#     response1 = tool.run("grass")
#     print(response1)
    
#     # Check stats
#     print(f"\n[Stats] Unique Pokemon so far: {tool.get_unique_stats()['total_unique_pokemon_retrieved']}")

#     print("\n>>> Agent Query 2: Search for 'Zarude' (Specific Name)")
#     response2 = tool.run("zarude")
#     print(response2)
    
#     # Check stats (Should increase if Zarude wasn't in the first result)
#     print(f"\n[Stats] Unique Pokemon so far: {tool.get_unique_stats()['total_unique_pokemon_retrieved']}")

#     print("\n>>> Agent Query 3: Search for 'Grass' again (Should not increase unique count for same items)")
#     response3 = tool.run("grass")
#     # print(response3) # Optionally print content
    
#     # The internal counter for unique docs shouldn't rise significantly (unless order changed and new items appeared in top 5)
#     print(f"\n[Stats] Final Unique Pokemon count: {tool.get_unique_stats()['total_unique_pokemon_retrieved']}")