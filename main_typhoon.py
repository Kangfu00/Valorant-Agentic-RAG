import os
from dotenv import load_dotenv

# 1. เปลี่ยน Library นำเข้าเป็น ChatOpenAI (รองรับ API ของ Typhoon)
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from tools import search_agent_strategy, search_map_strategy, search_weapon_strategy, search_update_patch, calculate_economy

load_dotenv()

# 2. โหลด API Key ของ Typhoon
typhoon_api_key = os.environ.get("TYPHOON_API_KEY")

def main():
    # 3. ตั้งค่า LLM เป็น Typhoon โดยชี้ URL ไปที่เซิร์ฟเวอร์ของ Typhoon
    llm = ChatOpenAI(
        api_key=typhoon_api_key,
        base_url="https://api.opentyphoon.ai/v1", 
        model="typhoon-v2.5-30b-a3b-instruct", # สามารถเปลี่ยนเป็นรุ่นล่าสุดที่เว็บระบุได้
        temperature=0, # ใช้ 0 เพื่อความแม่นยำจาก Tools
        max_tokens=3000
    )

    tools = [
        search_agent_strategy, 
        search_map_strategy, 
        search_weapon_strategy, 
        search_update_patch, 
        calculate_economy
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "คุณคือผู้ช่วยโค้ชเกม Valorant สุดอัจฉริยะ หน้าที่ของคุณคือตอบคำถามและให้คำแนะนำผู้เล่น "
                   "**กฎสำคัญในการทำงาน:**\n"
                   "1. หากคำถามมีหลายประเด็น (เช่น ถามเรื่องเงิน + ควรอีโค่หรือซื้อปืนอะไร) คุณ **ต้องเรียกใช้ Tool หลายตัวต่อเนื่องกัน** เช่น เรียก calculate_economy ก่อน แล้วตามด้วย search_weapon_strategy ทันที ห้ามคิดไปเอง\n"
                   "2. ห้ามแนะนำไอเทมจากเกมอื่นเด็ดขาด (เช่น ระเบิด Smoke/Flashbang แบบซื้อได้) สกิลใน Valorant ผูกกับตัวละครเท่านั้น\n"
                   "3. หาก Tool ขัดข้องหรือติดลิมิต ห้ามเดาคำตอบเอง ให้แจ้งผู้ใช้ว่าระบบขัดข้อง\n"
                   "4. ทับศัพท์ภาษาอังกฤษสำหรับคำศัพท์เฉพาะในเกมเสมอ (เช่น Force Buy, Eco, Light Shield, Outlaw)\n"
                   "จงวิเคราะห์คำถามของผู้ใช้อย่างละเอียด และเลือกใช้ปืนให้เหมาะสมกับงบประมาณและสถานการณ์เกราะของศัตรู"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"), 
    ])

    # สร้าง Agent และ Executor 
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # verbose=True เพื่อโชว์ Log การทำงานออก Terminal
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    print("=== 🌪️ เริ่มต้น Valorant Agentic RAG (Powered by Typhoon) ===")
    print("พิมพ์ 'exit' เพื่อออก\n")

    while True:
        user_input = input("👤 คุณ (ผู้เล่น): ")
        if user_input.lower() == 'exit':
            break
        
        # ใช้ try-except ป้องกัน Error เวลาเซิร์ฟเวอร์ API ทำงานหนัก
        try:
            response = agent_executor.invoke({"input": user_input})
            final_answer = response['output']
            
            # ตรวจสอบและประกอบร่างข้อความ
            if isinstance(final_answer, list):
                clean_text = ""
                for item in final_answer:
                    if isinstance(item, dict) and 'text' in item:
                        clean_text += item['text']
                    elif isinstance(item, str):
                        clean_text += item
                final_answer = clean_text

            print(f"\n🧠 Agent ตอบ:\n{final_answer}\n")
            print("-" * 50)
            
        except Exception as e:
            print(f"\n⚠️ เกิดข้อผิดพลาดในการเชื่อมต่อ: {e}")
            print("กรุณารอสักครู่แล้วพิมพ์ถามใหม่อีกครั้งครับ\n")
            print("-" * 50)

if __name__ == "__main__":
    main()