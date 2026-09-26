from dotenv import load_dotenv
load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter # split up our documents
from langchain_unstructured import UnstructuredLoader # to load a doc from web
from langchain_chroma import Chroma # for store our docs to vectorstore database
from langchain_openai import OpenAIEmbeddings # embed our docs

# Create a list of urls from where we get our articles
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load the docs and split them in chunks and embed them and store them to chroma
docs = [UnstructuredLoader(web_url=url, chunking_strategy="basic", max_characters=100000).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split out the docs into Chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
    persist_directory="./.chroma"
)

# create a retriever for docs
retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=OpenAIEmbeddings(),
).as_retriever()


