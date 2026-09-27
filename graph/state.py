from typing import List, TypedDict

class GraphState(TypedDict):
    """
    Represents the state of the graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: where to add search boolean flag
        documents: list of documents
    """
    question: str
    generation: str
    web_search: bool
    documents: List[str]

