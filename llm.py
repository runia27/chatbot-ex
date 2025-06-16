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
    

# 
def build_qa_prompt():
    ## [keyword dictionary]
    # 1. 기본형태 : 키 하나당 설명 하나, 단순하고 빠름
    # 용도 : FAQ 챗봇, 버튼식 응답
    # 2. 질문형태 : 유사한 질문에 여러 키로 분기하며 모두 같은 대답으로 연결, fallback 대응
    # 용도 : 단답 챗봇, 키워드 FAQ 챗봇
    # 3. 키워드 _ 태그 기반

    
    keyword_dictionary = {
#         '임대인': '''
# 임대인 또는 다음 각 목의 어느 하나에 해당하는 자를 말한다.
# 가. 임대인의 대리인, 그 밖에 임대인을 위하여 주택의 임대에 관하여 업무를 처리하는 자
# 나. 임대인의 의뢰를 받은 공인중개사(중개보조인을 포함한다)
# 다. 임대인을 위하여 임차인을 모집하는 자(그 피고용인을 포함한다)
# 라. 다수 임대인의 배후에 있는 동일인
# 마. 라목의 동일인이 지배하거나 경제적 이익을 공유하는 조직
# 바. 라목의 동일인이나 마목의 조직을 배후에 둔 다수의 임대인
#     ''',
#         '주택': '''
# 「주택임대차보호법」 제2조에 따른 주거용 건물(공부상 주거용 건물이 아니라도 임대차계약 체결 당시 임대차목적물의 구조와 실질이 주거용 건물이고 임차인의 실제 용도가 주거용인 경우를 포함한다)을 말한다.
#     ''',
#         '임대인 알려줘': '''
# 임대인은 다음과 같은 의미를 가집니다:

# 기본 정의:
# 임대인은 주택 등을 소유하고 이를 다른 사람에게 빌려주는 자를 의미합니다. 보통 임차인으로부터 임대료를 받고 주거 또는 사용을 허락합니다.

# 대리인 및 위임자:
# 임대인을 대신하여 임대 업무를 처리할 수 있는 대리인도 임대인의 범주에 포함됩니다.
# 임대인의 대리인은 주택의 임대에 관한 업무를 수행하는 사람입니다.

# 공인중개사 포함:
# 임대인의 의뢰를 받아 임차인과의 계약을 중개하는 공인중개사도 임대인의 범주에 들어갈 수 있습니다.

# 임차인 모집자:
# 임대인을 대신해 임차인을 모집하는 자도 포함됩니다. 이는 피고용인을 포함한 경우도 해당됩니다.

# 다수 임대인의 동일인 배후자:
# 여러 임대인을 뒤에서 지배하거나 경제적 이익을 공유하는 동일인도 임대인으로 간주될 수 있습니다.

# 조직 및 경제 공유:
# 동일인이 지배하는 조직 또는 경제적 이익을 공유하는 조직이 포함되며, 이들이 배후에 있는 다수의 임대인도 임대인으로 볼 수 있습니다.
# 이러한 내용은 특정 법적 상황에서 '임대인'의 정의가 어떻게 확장될 수 있는지를 보여줍니다. (전세사기피해자법)    
#     ''',    
    '임대인' : {
        'definition': '전세사기피해자법 제2조 2항에서 임대인을 정의합니다',
        'source': '전세사기피해자법 제2조',
        'tags': ['법률', '용어', '기초'],
        },
    '주택' : {
        'definition': '전세사기피해자법 제2조 1항에서 주택을 정의합니다',
        'source': '전세사기피해자법 제2조',
        'tags': ['법률', '용어', '기초'],
        },
    }

    dictionary_text = '\n'.join([
        f"{k} ({', '.join(v['tags'])}): {v['definition']} [출처: {v['source']}])" 
        for k, v in keyword_dictionary.items()
    ])

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

    print(f"대화이력: {get_session_history(session_id).messages} \n🐾\n")
    print('=' * 50)
    print(f"{session_id}")

    retriever = load_vectorstore().as_retriever(search_kwargs={'k': 2})
    search_result = retriever.invoke(user_message)
    print(f"\n검색결과 : \n{search_result}")

    return aimessage

