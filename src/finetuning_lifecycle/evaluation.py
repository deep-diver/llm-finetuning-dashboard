import random
from typing import Dict, Any, List

def compute_instruction_following_score(generation: str, instruction: str) -> float:
    """
    Mock method for evaluating formatting and length directives.
    In real usage, this would run heuristics or an LLM-as-a-judge score.
    """
    # Simple heuristics for tutorial stub:
    # 1. Did it output empty?
    if not generation.strip():
        return 0.0
    # 2. If instruction asks for JSON or specific formatting, does it match?
    if "json" in instruction.lower():
        if generation.strip().startswith("{") and generation.strip().endswith("}"):
            return 1.0
        return 0.2
    return 0.8 + random.random() * 0.2

def calculate_rouge_l_approx(gen: str, ref: str) -> float:
    """Computes basic word-level overlap approximation of ROUGE-L."""
    gen_words = set(gen.lower().split())
    ref_words = set(ref.lower().split())
    if not gen_words or not ref_words:
        return 0.0
    intersection = gen_words.intersection(ref_words)
    precision = len(intersection) / len(gen_words)
    recall = len(intersection) / len(ref_words)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)
