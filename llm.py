import os

from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
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


store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
    

# RetrievalQA 함수 정의
def get_retrievalQA():

    database = get_database()
    llm = get_llm()
    
    prompt = ('''
[Identity]
- 당신은 전세사기 피해 법률 전문가입니다. 
- "context"를 참고해 질문에 대해 일목요연하고 구체적으로 답변하세요. 
- 항목별로 표시해서 답변해주세요. 
- 답변 마지막 부분에는 관련 법조항을 소괄호 안에 넣어 같이 명시해주세요. 
- 어떤 것에 대해 알려달라는 질문은 어떤 것에 대해 설명해달라는 말과 같습니다. 
- 사용자를 직접적으로 언급하지 않습니다. 
- 관련되지 않은 질문에는 전세사기와 관련된 비슷한 질문을 물어본게 맞는지 되묻습니다.

context: {context}       
question: {input}                                   
answer:
    ''')


    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return '\n\n'.join(doc.page_content for doc in docs)
    
    inputText = RunnableLambda(lambda x: x["input"])

    chain_lcel = (
        {
            "context" : inputText | database.as_retriever() | format_docs,
            "input" : inputText,
            "chat_history" : RunnableLambda(lambda x: x["chat_history"])
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )    

    all_chain = RunnableWithMessageHistory(
        chain_lcel,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return all_chain


## AI 메세지 함수 정의 
def get_aimessage(user_message, session_id='default'):        
    all_chain = get_retrievalQA()

    aimessage = all_chain.invoke(
        {"input": user_message},
        config={"configurable":{"session_id": session_id}},
    )
    return aimessage
