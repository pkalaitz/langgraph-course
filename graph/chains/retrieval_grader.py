from langchain_core.prompts import ChatPromptTemplate # we want a structured answer systemmessage, humanmessage etc
from pydantic import BaseModel, Field # we want a pydantic object with info of yes/no relevant
from langchain_openai import ChatOpenAI # for LLM chain

llm = ChatOpenAI(model="gpt-6-astra")  # initialized with model gpt-3.5-turbo but id doesn't accept structured output but JSON mode
# tested with model gpt-4o-mini but works with astra too
# developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents"""

    binary_score: str = Field(
        description="Documents are relevant to question, 'yes' or 'no'. "
    )

structured_llm_grader = llm.with_structured_output(GradeDocuments) #the llm is going to leverage the description in order to discern yes/no for doc

system = """You are a grader assesing relevance of a retrieved document to a user question. \n
        if the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader # this is the chain grade_prompt--> LLM --> structured output


