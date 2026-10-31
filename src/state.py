from typing import List, TypedDict, Any, Dict
from typing import List, Dict, Any, TypedDict

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question asked by the user
        generation: LLM generation result
        documents: list of retrieved documents (dictionaries with content & source)
        router_decision: routing choice ("generate" or "web_search")
    """
    question: str
    generation: str
    documents: List[Dict[str, Any]]
    router_decision: str