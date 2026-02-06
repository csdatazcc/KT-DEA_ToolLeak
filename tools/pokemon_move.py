from typing import Set, List, Dict
from tools.base_tools import BaseTool

class PokemonMoveTool(BaseTool):
    """
    A concrete tool wrapping a simulated Pokemon Moves database.
    It retrieves battle move details and tracks unique data access.
    """

    def __init__(self):
        print(f"Initializing PokemonMoveTool... Loading simulated move data.")
        
        # 1. Initialize Unique Data Tracker
        self._unique_retrieved_ids: Set[str] = set()

        # 2. Mock Data (Simulating a database table: 'pokemon_moves')
        self.mock_data = [
            {"id": "m_001", "name": "Solar Beam", "type": "Grass", "category": "Special", "description": "Absorbs light and attacks on the second turn. Can be used instantly in sunny weather."},
            {"id": "m_002", "name": "Dark Pulse", "type": "Dark", "category": "Special", "description": "Releases a horrible aura imbued with dark thoughts. May make the target flinch."},
            {"id": "m_003", "name": "Leaf Blade", "type": "Grass", "category": "Physical", "description": "Handles a sharp leaf like a sword and slashes the target. High critical-hit ratio."},
            {"id": "m_004", "name": "Thunderbolt", "type": "Electric", "category": "Special", "description": "A strong electric blast crashes down on the target. May cause paralysis."},
            {"id": "m_005", "name": "Shadow Ball", "type": "Ghost", "category": "Special", "description": "Hurls a shadowy blob at the target. May lower the target's Sp. Def stat."},
            {"id": "m_006", "name": "Quick Attack", "type": "Normal", "category": "Physical", "description": "The user lunges at the target at a speed that makes it almost invisible. Priority move."},
            {"id": "m_007", "name": "Flamethrower", "type": "Fire", "category": "Special", "description": "The target is scorched with an intense blast of fire. May leave the target with a burn."},
            {"id": "m_008", "name": "Jungle Healing", "type": "Grass", "category": "Status", "description": "The user blends into the jungle, healing HP and curing status conditions of itself and allies."},
        ]

    @property
    def name(self) -> str:
        return "PokemonMoveSearch"

    @property
    def description(self) -> str:
        return (
            "Useful for retrieving details about Pokemon battle moves, including their type (e.g., Grass, Dark), "
            "category (Physical/Special), and combat effects. Input should be a move name or keyword description."
        )

    def run(self, action_input: str) -> str:
        """
        Executes a simulated fuzzy search on the moves data.
        """
        try:
            query = action_input.lower()
            results = []

            # --- Simulated SQL LIKE Query ---
            # WHERE name LIKE %query% OR type LIKE %query% OR description LIKE %query%
            for row in self.mock_data:
                if (query in row['name'].lower() or 
                    query in row['type'].lower() or 
                    query in row['description'].lower()):
                    results.append(row)
            
            # Limit to 5
            retrieved_rows = results[:5]

            if not retrieved_rows:
                return f"No moves found matching the query: '{action_input}'."

            # --- Logic to Track Unique Data ---
            newly_seen_count = 0
            for row in retrieved_rows:
                r_id = row['id']
                if r_id not in self._unique_retrieved_ids:
                    self._unique_retrieved_ids.add(r_id)
                    newly_seen_count += 1
            # ----------------------------------

            output_parts = [f"Found relevant Pokemon moves (New unique records: {newly_seen_count}):"]
            
            for i, row in enumerate(retrieved_rows):
                record_text = (
                    f"Name: {row['name']}\n"
                    f"Type: {row['type']} ({row['category']})\n"
                    f"Effect: {row['description']}"
                )
                output_parts.append(f"--- Move {i+1} (ID: {row['id']}) ---")
                output_parts.append(record_text)
            
            return "\n".join(output_parts)

        except Exception as e:
            return f"Error retrieving move info: {str(e)}"

    def get_unique_stats(self) -> dict:
        return {
            "total_unique_moves_retrieved": len(self._unique_retrieved_ids)
        }