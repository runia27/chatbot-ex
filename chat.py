import streamlit as st
from llm import get_aimessage


st.set_page_config(page_title='전세사기피해 상담 챗봇', page_icon='🤖')
st.title('전세사기피해 상담 챗봇 🤖')

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
        aimessage = get_aimessage(inputchat)
        
        with st.chat_message("ai"):
            st.write(aimessage)
        st.session_state.message_list.append({'role': 'ai', 'content': aimessage})


