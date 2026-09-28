#from langchain_classic import hub # we want to import the prompt from it, but old and dangerous version
# initial edenmarco from langchain import hub
from dotenv import load_dotenv
load_dotenv()

from langchain_core.output_parsers import StrOutputParser # get the content of our message and turn it to a string
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-6-astra") # attention astra doesn't need temperature=0.0
"""
prompt = hub.pull("rlm/rag-prompt") # this formal in ChatPromptTemplate give to the llm the role of an assistant rag-prompt
                                    # very standard RAG prompt
"""

# Inline equivalent of hub.pull("rlm/rag-prompt") - the hub module was removed
# from the langchain package, and pulling public prompts now requires a
# LangSmith API key.

prompt = ChatPromptTemplate.from_messages([
    (
        "human",
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise. \n"
        "Question: {question} \nContext: {context} \nAnswer: ",
    )
])

generation_chain = prompt | llm | StrOutputParser()  # the pipe is going prompt-->llm-->StrOutputParser


