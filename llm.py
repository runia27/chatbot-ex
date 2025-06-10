import os

from dotenv import load_dotenv
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone



## AI 메세지 함수 정의 
def get_aimessage(user_message):

    load_dotenv()
    api_key = os.getenv('PINECONE_API_KEY')
    LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

    pc = Pinecone(api_key=api_key)

    ## 임베딩
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-large",
    )

    # 파인콘에 저장 
    
    database = PineconeVectorStore.from_existing_index(
        embedding=embedding,
        index_name='law1and2-quiz4'
    )

    llm = ChatOpenAI()

    prompt = hub.pull("rlm/rag-prompt")

    ## RetrievalQA 구현
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

    aimessage = chain_lcel.invoke(user_message)
    return aimessage




