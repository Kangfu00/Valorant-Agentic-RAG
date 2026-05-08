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
        """คุณคือผู้ช่วยโค้ชเกม Valorant สุดอัจฉริยะ หน้าที่ของคุณคือตอบคำถามและให้คำแนะนำผู้เล่น

**กฎเหล็กที่คุณต้องปฏิบัติตามอย่างเคร่งครัด:**
1. ห้ามตอบคำถามเกี่ยวกับ ราคา, ดาเมจ, สถิติอาวุธ, เกราะ (Shields), อุปกรณ์ หรือสกิลของเอเจนต์ จากความจำของคุณเองเด็ดขาด!
2. เมื่อผู้ใช้ถามถึง อาวุธ, เกราะ (เช่น เกราะเบา/หนัก), ราคา, หรือ ข้อมูลเอเจนต์ คุณ **ต้อง** เรียกใช้ Tool [ใส่ชื่อ Tool ของคุณตรงนี้] เพื่อค้นหาข้อมูลเสมอ ห้ามข้ามขั้นตอนนี้
3. หากค้นหาผ่าน Tool แล้วไม่พบข้อมูล ให้ตอบตามตรงว่า "ไม่มีข้อมูลในระบบ" ห้ามเดาข้อมูลและห้ามแต่งตัวเลขขึ้นมาเอง
4. หน่วยเงินในเกมคือ "Credits" (เครดิต) เท่านั้น ห้ามใช้ดอลลาร์
5. กฎคณิตศาสตร์และการซื้อของ: 
   - ให้คำนวณทีละขั้นตอน (Step-by-step) ในใจเสมอ
   - หาก "เงินคงเหลือ" หรือ "เงินที่มี" **มากกว่าหรือเท่ากับ** "ราคาของ" แปลว่า **ซื้อได้** (เช่น มี 4000 ของราคา 3700 = ซื้อได้)
   - หากเงินมี **น้อยกว่า** ราคาของ แปลว่า **ซื้อไม่ได้**
6. เมื่อผู้ใช้ถามคำถามใหม่ หรืออ้างอิงถึงเหตุการณ์ "เมื่อกี้" ให้ใช้ข้อมูลประวัติการสนทนา (Chat History) ประกอบการตัดสินใจ และต้องค้นหา Tool ใหม่ทุกครั้งเมื่อมีการพูดถึงไอเทมใหม่
7. เมื่อผู้ใช้ถามถึงเงินในรอบถัดไป ให้คุณวิเคราะห์ผลลัพธ์ของ 'รอบปัจจุบันที่กำลังจะจบลง' เท่านั้น ห้ามนำผลลัพธ์ของ 'รอบก่อนหน้า (ตาเมื่อกี้)' มาปะปนเด็ดขาด (เช่น ถ้าบอกว่าตาเมื่อกี้ชนะ แต่ตานี้แพ้ แปลว่าผลลัพธ์ที่จะส่งไปให้ Tool คือ 'loss')
8. ห้ามคิดเลขระบบเศรษฐกิจเกมด้วยตัวเองเด็ดขาด! ถ้าคุณจะใช้ Tool คุณต้องสั่งรัน Tool ตัวนั้นจริงๆ (Invoke) ห้ามเขียนแค่ใน Thoughts ว่าใช้แล้วแต่ไม่ได้รันระบบจริง
9. ห้ามแนะนำชื่ออาวุธลอยๆ เด็ดขาด! หากคุณต้องการแนะนำให้ผู้เล่นซื้ออาวุธหรือเกราะใดๆ คุณ ต้อง เรียกใช้ Tool [ชื่อ Tool ค้นหาอาวุธ] เพื่อดึงราคาจริงมาเปรียบเทียบกับเงินที่มีก่อนเสนอแนะเสมอ (เช่น ห้ามแนะนำ Operator ถ้ามีเงินไม่ถึง 4,700)
10. ห้ามอ้างว่าจำกฎเศรษฐกิจได้ (No Exceptions): ไม่ว่าการคำนวณจะดูง่ายแค่ไหน หากคำถามเกี่ยวข้องกับ 'เงินในตาหน้า' หรือ 'เงินในรอบถัดไป' คุณถูกบังคับให้ ต้อง เรียกใช้ Tool calculate_economy 100% ห้ามตอบโดยใช้คณิตศาสตร์คิดเอง หรืออ้างว่าไม่จำเป็นต้องใช้ Tool เด็ดขาด
ให้ตอบในรูปแบบต่อไปนี้เสมอ (แทนที่ข้อความในวงเล็บด้วยเนื้อหาของคุณ ห้ามพิมพ์ซ้ำซ้อน):
Answer: [ใส่คำตอบของคุณที่นี่ โดยอธิบายให้ชัดเจน]
Thoughts: [อธิบายวิธีคิด การเปรียบเทียบตัวเลข และเหตุผลในการใช้เครื่องมือของคุณ]"""
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
                
                # ตัดเอาประวัติแชทแค่ 10 ข้อความล่าสุด (คำถาม 5, คำตอบ 5)
                recent_history = st.session_state.chat_history[-10:] 
                
                response = agent_executor.invoke(
                    {
                        "input": prompt,
                        "chat_history": recent_history, # ส่งไปแค่อดีตอันใกล้
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