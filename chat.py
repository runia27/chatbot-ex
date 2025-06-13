import uuid
import streamlit as st
from llm import stream_get_aimessage


st.set_page_config(page_title='전세사기피해 상담 챗봇', page_icon='💒')
st.title('전세사기피해 상담 챗봇 💒')


# 세션 상태에 'session_id'가 없으면 고유한 UUID를 생성해 
# 같은 탭에서는 새로고침해도 동일한 session_id 유지
# if 'session_id' not in st.session_state:
#     st.session_state.session_id = str(uuid.uuid4())

# URL에 session_id가 있으면 우선 사용하고 없으면 만들어서 붙임. 이후 같은 탭에서 유지
if 'session_id' in st.query_params:
    session_id = st.query_params.session_id
else:
    session_id = str(uuid.uuid4())
    st.query_params.update({"session_id":session_id})

st.session_state.session_id = session_id


# 세션 상태에 'message_list'가 없으면 빈 리스트로 초기화함
if 'message_list' not in st.session_state:
    st.session_state.message_list = []

## 이전 채팅 화면 출력
for i in st.session_state.message_list:
    with st.chat_message(i['role']):
        st.write(i['content'])

## 채팅창 및 대화내역 구현 
if inputchat := st.chat_input(placeholder="전세사기 피해와 관련된 질문을 작성해 주세요."):
    with st.chat_message("user"):    
        st.write(f"{inputchat}")
    st.session_state.message_list.append({'role': 'user', 'content': inputchat})
    
    # 답변 나올 때까지 스피너 
    with st.spinner("답변을 생성하는 중입니다."):
        session_id = st.session_state.session_id
        aimessage = stream_get_aimessage(inputchat, session_id=session_id)
        
        with st.chat_message("ai"):
            aimessage = st.write_stream(aimessage)
        st.session_state.message_list.append({'role': 'ai', 'content': aimessage})

