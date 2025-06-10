import os

import streamlit as st
from dotenv import load_dotenv
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone


st.set_page_config(page_title='전세사기피해 상담 챗봇', page_icon='🤖')

st.title('전세사기피해 상담 챗봇 🤖')


if 'message_list' not in st.session_state:
    st.session_state.message_list = []

print(f"before: {st.session_state.message_list}")

## 이전 채팅 화면 출력
for i in st.session_state.message_list:
    with st.chat_message(i['role']):
        st.write(i['content'])


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

## 프롬프트창(채팅창) 구현
if inputchat := st.chat_input(placeholder="전세사기 피해와 관련된 질문을 작성해 주세요."):
    with st.chat_message("user"):    
        st.write(f"{inputchat}")
    st.session_state.message_list.append({'role': 'user', 'content': inputchat})
    
    aimessage = get_aimessage(inputchat)
    
    with st.chat_message("ai"):
        st.write(aimessage)
    st.session_state.message_list.append({'role': 'ai', 'content': aimessage})


print(f"after: {st.session_state.message_list}")