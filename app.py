import os
import time
import io
import sys
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from tools import (
    search_agent_strategy,
    search_map_strategy,
    search_weapon_strategy,
    search_update_patch,
    calculate_economy,
)

load_dotenv()

# --- 1. การตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Valorant Assistant (Typhoon)", page_icon="🌀")
st.title("🌀 Valorant AI Assistant")
st.caption("Powered by Typhoon v2.5")

# --- 2. ตั้งค่า Typhoon API ---
TYPHOON_API_KEY = os.environ.get("TYPHOON_API_KEY", "")
if not TYPHOON_API_KEY:
    st.error("กรุณาตั้งค่า TYPHOON_API_KEY ในตัวแปรแวดล้อมหรือไฟล์ .env แล้วรีเฟรชหน้า")
    st.stop()

system_prompt = (
    "คุณคือผู้ช่วยโค้ชเกม Valorant สุดอัจฉริยะ หน้าที่ของคุณคือตอบคำถามและให้คำแนะนำผู้เล่น\n"
    "**กฎสำคัญ:**\n"
    "1. เรียกใช้ Tool หลายตัวต่อเนื่องกันหากจำเป็น\n"
    "2. ห้ามแนะนำไอเทมจากเกมอื่น สกิลผูกกับตัวละครเท่านั้น\n"
    "3. ห้ามเดาคำตอบเอง หากไม่พบข้อมูลให้แจ้งระบบขัดข้อง\n"
    "4. ทับศัพท์ภาษาอังกฤษเสมอ (เช่น Force Buy, Eco)\n"
    "5. หากไม่มีข้อมูล ให้ตอบว่าไม่พบข้อมูลอย่างชัดเจน\n"
    "ให้ตอบในรูปแบบต่อไปนี้:\n"
    "Answer: <คำตอบ>\n"
    "Thoughts: <สิ่งที่เอไอคิดหรือวิธีค้นหาข้อมูล>"
)

tools = [
    search_agent_strategy,
    search_map_strategy,
    search_weapon_strategy,
    search_update_patch,
    calculate_economy,
]

prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        "คุณคือผู้ช่วยโค้ชเกม Valorant สุดอัจฉริยะ หน้าที่ของคุณคือตอบคำถามและให้คำแนะนำผู้เล่น\n"
        "**กฎสำคัญ:**\n"
        "1. เรียกใช้ Tool หลายตัวต่อเนื่องกันหากจำเป็น\n"
        "2. ห้ามแนะนำไอเทมจากเกมอื่น สกิลผูกกับตัวละครเท่านั้น\n"
        "3. ห้ามเดาคำตอบเอง หากไม่พบข้อมูลให้แจ้งระบบขัดข้อง\n"
        "4. ทับศัพท์ภาษาอังกฤษเสมอ (เช่น Force Buy, Eco)\n"
        "5. หากไม่มีข้อมูล ให้ตอบว่าไม่พบข้อมูลอย่างชัดเจน\n"
        "ให้ตอบในรูปแบบต่อไปนี้:\n"
        "Answer: <คำตอบ>\n"
        "Thoughts: <สิ่งที่เอไอคิดหรือวิธีค้นหาข้อมูล>"
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


def build_agent_executor():
    llm = ChatOpenAI(
        api_key=TYPHOON_API_KEY,
        base_url="https://api.opentyphoon.ai/v1",
        model="typhoon-v2.5-30b-a3b-instruct",
        temperature=0,
        max_tokens=3000,
    )

    agent = create_tool_calling_agent(llm, tools, prompt_template)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=25,
        handle_parsing_errors=True,
    )


# --- 3. จัดการประวัติการสนทนา ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("thoughts"):
            with st.expander("ดูสิ่งที่เอไอคิด", expanded=False):
                st.code(message["thoughts"], language="text")


def parse_ai_response(response_text: str) -> tuple[str, str]:
    answer = response_text
    thoughts = ""

    if "Thoughts:" in response_text:
        answer_part, thoughts_part = response_text.split("Thoughts:", 1)
        answer = answer_part.replace("Answer:", "").strip()
        thoughts = thoughts_part.strip()
    elif "Answer:" in response_text:
        answer = response_text.split("Answer:", 1)[1].strip()

    return answer, thoughts


# --- 4. รับคำถามจากผู้ใช้ ---
if prompt := st.chat_input("ถามคำถามเกี่ยวกับ Valorant..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("🌪️ Typhoon กำลังคิด..."):
                # Capture stdout
                old_stdout = sys.stdout
                sys.stdout = captured_output = io.StringIO()
                
                agent_executor = build_agent_executor()
                response = agent_executor.invoke(
                    {
                        "input": prompt,
                        "chat_history": st.session_state.chat_history,
                    }
                )
                
                # Restore stdout
                sys.stdout = old_stdout
                log_output = captured_output.getvalue()
            
            final_answer = response["output"] if isinstance(response, dict) and "output" in response else str(response)
            if isinstance(final_answer, list):
                final_answer = "".join(
                    [item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in final_answer]
                )
            
            answer, thoughts = parse_ai_response(final_answer)
            st.markdown(answer)
            if log_output.strip():
                with st.expander("ดูสิ่งที่เอไอคิด", expanded=False):
                    st.code(log_output.strip(), language="text")
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "thoughts": log_output.strip(),
            })
            st.session_state.chat_history.append(HumanMessage(content=prompt))
            st.session_state.chat_history.append(AIMessage(content=answer))
        
        except Exception as e:
            sys.stdout = old_stdout if 'old_stdout' in locals() else sys.stdout  # restore in case of error
            st.error(f"Error: {e}")