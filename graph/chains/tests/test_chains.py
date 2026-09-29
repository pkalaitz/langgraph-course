from dotenv import load_dotenv

load_dotenv()

from pprint import pprint
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.generation import generation_chain  # this is from our file graph->chains->generation.py
from graph.chains.hallucination_grader import hallucination_grader, GradeHallucinations
from ingestion import retriever


def test_retrival_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "yes"


def test_retrival_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "how to make pizaa", "document": doc_txt}
    )

    assert res.binary_score == "no"


def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)

def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"context": docs, "question": question})

    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs,
         "generation": generation}
    )
    assert res.binary_score

def hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    # no generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs,
         "generation": "In order to make pizza we need to first start with dough"}
    )

    assert res.binary_score

