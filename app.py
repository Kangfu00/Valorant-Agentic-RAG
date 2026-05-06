import streamlit as st
from openai import OpenAI
import time

# --- 1. การตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Valorant Assistant (Typhoon)", page_icon="🌀")
st.title("🌀 Valorant AI Assistant")
st.caption("Powered by Typhoon v2.5")

# --- 2. ตั้งค่า Typhoon API ---
# ตรวจสอบ API Key ให้ถูกต้อง
TYPHOON_API_KEY = "sk-bWF7fiukGL182o56xYJoggBKjkCYGn6dgd6w8s7oS6Rlsj6z" 
client = OpenAI(
    base_url="https://api.opentyphoon.ai/v1",
    api_key=TYPHOON_API_KEY
)

# --- 3. จัดการประวัติการสนทนา ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. รับคำถามจากผู้ใช้ ---
if prompt := st.chat_input("ถามคำถามเกี่ยวกับ Valorant..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- 5. การตอบกลับของ AI ---
    with st.chat_message("assistant"):
        with st.status("🌪️ Typhoon กำลังคิด...", expanded=True) as status:
            st.write("🔍 กำลังค้นหาข้อมูลจากไฟล์ Markdown...")
            time.sleep(0.5)
            
            try:
                # แก้ไขชื่อโมเดลตามรูปภาพที่คุณส่งมา
                stream = client.chat.completions.create(
                    model="typhoon-v2.5-30b-a3b-instruct", 
                    messages=[
                        {"role": "system", "content": "You are an AI assistant named Typhoon created by SCB 10X. You are an expert in Valorant."},
                        {"role": "user", "content": prompt}
                    ],
                    stream=True,
                )
                status.update(label="ประมวลผลสำเร็จ!", state="complete", expanded=False)
                
                # แสดงผล Streaming
                response_placeholder = st.empty()
                full_response = ""
                
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content is not None:
                        full_response += content
                        response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                status.update(label="เกิดข้อผิดพลาด!", state="error")
                st.error(f"Error: {e}")