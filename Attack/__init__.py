"""
Attack package initializer.

This package provides:
- Structured keyword extraction utilities (extract.py)
- Adversarial stealing instruction generation (generate.py)
- Tool semantic clustering & selection utilities (TCL.py)
- Unified exporting of LLM classes
"""

# ------------------------
# Re-export LLM classes
# ------------------------
from llms import OllamaLLM, OpenAILLM, DeepseekLLM

# ------------------------
# extract.py exports
# ------------------------
from Attack.extract import (
    extract_system_prompt,
    parse_to_dict,
    keyword_extra,
    keyword_base_update,
)

# ------------------------
# generate.py exports
# ------------------------
from Attack.generate import (
    attack_system_prompt,
    attack_prompt_generate,
)

# ------------------------
# TCL.py exports
# ------------------------
from Attack.TCL import (
    expand_similar_tools,
    Relevant_Tool_Selection,
    PURPLE,
    CYAN,
    BRIGHT_GREEN,
    BRIGHT_YELLOW,
    BRIGHT_RED,
    WHITE,
    RESET,
)

from Attack.key_word_v2 import ToolSemanticProcessor

# ------------------------
# key_word_v2 exports
# ------------------------


__all__ = [
    # LLMs
    "OllamaLLM",
    "OpenAILLM",
    "DeepseekLLM",

    # extract.py
    "extract_system_prompt",
    "parse_to_dict",
    "keyword_extra",
    "keyword_base_update",

    # generate.py
    "attack_system_prompt",
    "attack_prompt_generate",

    # TCL.py utilities
    "expand_similar_tools",
    "Relevant_Tool_Selection",

    # Color constants
    "PURPLE",
    "CYAN",
    "BRIGHT_GREEN",
    "BRIGHT_YELLOW",
    "BRIGHT_RED",
    "WHITE",
    "RESET",

    # key_word_v2
    "ToolSemanticProcessor",
]
