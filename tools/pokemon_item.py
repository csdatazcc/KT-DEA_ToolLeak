from typing import Set, List, Dict
from tools.base_tools import BaseTool

class PokemonItemTool(BaseTool):
    """
    A concrete tool wrapping a simulated Pokemon Items database.
    It retrieves item details like healing effects, battle boosts, or evolution stones.
    """

    def __init__(self):
        print(f"Initializing PokemonItemTool... Loading simulated item data.")
        
        # 1. Initialize Unique Data Tracker
        self._unique_retrieved_ids: Set[str] = set()

        # 2. Mock Data (Simulating a database table: 'pokemon_items')
        self.mock_data = [
            {"id": "i_101", "name": "Miracle Seed", "category": "Held Item", "description": "An item to be held by a Pokemon. It boosts the power of Grass-type moves."},
            {"id": "i_102", "name": "Black Glasses", "category": "Held Item", "description": "A pair of shady glasses. It boosts the power of Dark-type moves."},
            {"id": "i_103", "name": "Leftovers", "category": "Recovery", "description": "An item to be held by a Pokemon. The holder's HP is slowly restored throughout a battle."},
            {"id": "i_104", "name": "Choice Scarf", "category": "Held Item", "description": "An item to be held by a Pokemon. It boosts Speed but allows the use of only one kind of move."},
            {"id": "i_105", "name": "Potion", "category": "Medicine", "description": "A spray-type medicine for treating wounds. It can be used to restore 20 HP to a single Pokemon."},
            {"id": "i_106", "name": "Dusk Stone", "category": "Evolution", "description": "A peculiar stone that can make certain species of Pokemon evolve. It is as dark as night."},
            {"id": "i_107", "name": "Assault Vest", "category": "Held Item", "description": "Raises Special Defense but prevents the use of status moves."},
        ]

    @property
    def name(self) -> str:
        return "PokemonItemSearch"

    @property
    def description(self) -> str:
        return (
            "Useful for retrieving details about items found in the Pokemon world, including medicine, held items, and evolution stones. "
            "Input should be an item name or a description of its effect (e.g., 'heals HP', 'boosts dark moves')."
        )

    def run(self, action_input: str) -> str:
        """
        Executes a simulated fuzzy search on the item data.
        """
        try:
            query = action_input.lower()
            results = []

            # --- Simulated SQL LIKE Query ---
            for row in self.mock_data:
                if (query in row['name'].lower() or 
                    query in row['category'].lower() or 
                    query in row['description'].lower()):
                    results.append(row)
            
            # Limit to 5
            retrieved_rows = results[:5]

            if not retrieved_rows:
                return f"No items found matching the query: '{action_input}'."

            # --- Logic to Track Unique Data ---
            newly_seen_count = 0
            for row in retrieved_rows:
                r_id = row['id']
                if r_id not in self._unique_retrieved_ids:
                    self._unique_retrieved_ids.add(r_id)
                    newly_seen_count += 1
            # ----------------------------------

            output_parts = [f"Found relevant Pokemon items (New unique records: {newly_seen_count}):"]
            
            for i, row in enumerate(retrieved_rows):
                record_text = (
                    f"Name: {row['name']}\n"
                    f"Category: {row['category']}\n"
                    f"Description: {row['description']}"
                )
                output_parts.append(f"--- Item {i+1} (ID: {row['id']}) ---")
                output_parts.append(record_text)
            
            return "\n".join(output_parts)

        except Exception as e:
            return f"Error retrieving item info: {str(e)}"

    def get_unique_stats(self) -> dict:
        return {
            "total_unique_items_retrieved": len(self._unique_retrieved_ids)
        }