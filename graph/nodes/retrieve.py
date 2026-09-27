from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever

# Make the node retrieve
def retrieve(state: GraphState)->Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]

    documents = retriever.invoke(question)
    return {"documents" : documents, "question" : question}



