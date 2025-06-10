import os

from dotenv import load_dotenv
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

# llm 함수 정의
def get_llm(model="gpt-4o"):
    llm = ChatOpenAI(model=model)
    return llm


# 데이터베이스 함수 정의 
def get_database():
    api_key = os.getenv('PINECONE_API_KEY')
    Pinecone(api_key=api_key)

    embedding = OpenAIEmbeddings(
        model="text-embedding-3-large",
    )    
   
    database = PineconeVectorStore.from_existing_index(
        embedding=embedding,
        index_name="law1and2-quiz4",
    )
    return database


# RetrievalQA 함수 정의
def get_retrievalQA():
    LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

    database = get_database()
    llm = get_llm()
    prompt = hub.pull(
        "moon/rag-prompt",
        api_key=LANGCHAIN_API_KEY,
    )


    def format_docs(docs):
        return '\n\n'.join(doc.page_content for doc in docs)

    chain_lcel = (
        {
            "context" : database.as_retriever() | format_docs,
            "question" : RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain_lcel


## AI 메세지 함수 정의 
def get_aimessage(user_message):        
    chain_lcel = get_retrievalQA()

    aimessage = chain_lcel.invoke(user_message)
    return aimessage
