from typing import List, TypedDict

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question asked by the user
        generation: LLM generation result
        documents: list of retrieved documents (from Chroma or Tavily)
    """
    question: str
    generation: str
    documents: List[str]