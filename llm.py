import json
import os

from dotenv import load_dotenv
from langchain.chains import (create_history_aware_retriever,
                              create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import (ChatPromptTemplate, FewShotPromptTemplate,
                                    MessagesPlaceholder, PromptTemplate)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from config import answer_examlples


load_dotenv()

# llm 함수 정의
def load_llm(model="gpt-4o"):
    llm = ChatOpenAI(model=model)
    return llm


# 데이터베이스 함수 정의 
def load_vectorstore():
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

# 세션 아이디 생성
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# 히스토리 기반 리트리버
def build_history_aware_retriever(llm, retriever):

    question_prompt = ('''
- 사용자의 질문이 이전 대화 맥락을 참조한다면, 이를 바탕으로 누구나 이해할 수 있도록 질문을 완전한 문장으로 재작성합니다.
- 질문이 이미 독립적인 문장이라면 그대로 반환합니다.
- 답변은 하지 않고, 오직 질문만 출력합니다.
- 항상 질문 형태로 끝나야 하며, 문장은 정확하고 명확해야 합니다.

''')

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )
    return history_retriever


## few shot
def build_few_shot_examples() -> str:

    example_prompt = PromptTemplate.from_template("Question: {input}\n{answer}")

    few_shot_prompt = FewShotPromptTemplate(
        examples=answer_examlples,
        example_prompt=example_prompt,
        prefix= "다음 질문에 답변하세요 : ",
        suffix="Question: {input}",
        input_variables=["input"],
    )
   
    formated_few_shot_prompt = few_shot_prompt.format(input="{input}")
    return formated_few_shot_prompt
    

## [keyword dictionary]
# 1. 기본형태 : 키 하나당 설명 하나, 단순하고 빠름
# 용도 : FAQ 챗봇, 버튼식 응답
# 2. 질문형태 : 유사한 질문에 여러 키로 분기하며 모두 같은 대답으로 연결, fallback 대응
# 용도 : 단답 챗봇, 키워드 FAQ 챗봇 
# 3. 키워드 _ 태그 기반 // test.py 파일에서 확인. 
# 4. 외부 파일로 가져오기(json)
def load_dictionary_from_file(path='keyword_dictionary.json'):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file) 
    
## 문자열로 
def bulid_dictionay_text(dictionary: dict) -> str:

    dictionary_text = '\n'.join([
        f"{k} ({', '.join(v['tags'])}): {v['definition']} [출처: {v['source']}])" 
        for k, v in dictionary.items()
    ])    
    return dictionary_text

    
# 
def build_qa_prompt():
    keyword_dictionary = load_dictionary_from_file()    
    dictionary_text = bulid_dictionay_text(keyword_dictionary)
    print(dictionary_text)

    prompt = ('''
[Identity]
- 당신은 전세사기 피해 법률 전문가입니다. 
- "context"와 "keyward_dictionary"를 참고해 질문에 대해 일목요연하고 구체적으로 답변하세요. 
- 항목별로 표시해서 답변해주세요. 
- 답변 마지막 부분에는 관련 법조항을 소괄호 안에 넣어 같이 명시해주세요. 
- 어떤 것에 대해 알려달라는 질문은 어떤 것에 대해 설명해달라는 말과 같습니다. 
- 사용자를 직접적으로 언급하지 않습니다. 
- 관련되지 않은 질문에는 전세사기와 관련된 비슷한 질문을 물어본게 맞는지 되묻습니다.

context: {context}       
question: {input}    
keyword_dictionary: {dictionary_text}                               
answer:
    ''')

    formated_few_shot_prompt = build_few_shot_examples()

    qa_prompt = ChatPromptTemplate.from_messages([
    ("system", prompt),
    ("assistant", formated_few_shot_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    ]).partial(dictionary_text=dictionary_text)
    
    return qa_prompt


# RetrievalQA 함수 정의
def get_all_chain():

    database = load_vectorstore()
    retriever = database.as_retriever(search_kwargs={'k': 2})
    llm = load_llm()

    history_retriever = build_history_aware_retriever(llm, retriever)
    qa_prompt = build_qa_prompt()
    
    answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_retriever, answer_chain)

    all_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    ).pick("answer")

    return all_chain


## AI 메세지 함수 정의 
def stream_get_aimessage(user_message, session_id='default'):        
    all_chain = get_all_chain()

    aimessage = all_chain.stream(
        {"input": user_message},
        config={"configurable":{"session_id": session_id}},
    )

    return aimessage

