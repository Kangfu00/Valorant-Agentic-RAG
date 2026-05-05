import os
from dotenv import load_dotenv
# 1. เปลี่ยน Library นำเข้าเป็นของ Google GenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from tools import search_agent_strategy, search_map_strategy, search_weapon_strategy, search_update_patch, calculate_economy
from langchain_community.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder

load_dotenv(override=True)
# โหลด API Key
# 2. เปลี่ยนชื่อตัวแปรเป็น GOOGLE_API_KEY (ไปสร้างคีย์ฟรีได้ที่ Google AI Studio)
api_key = os.environ.get("GOOGLE_API_KEY")

def main():
    # 3. ตั้งค่า LLM เป็น Gemini (แนะนำ gemini-1.5-flash เพราะเร็วและรองรับ Tool Calling ได้ดีมาก)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0 # ใช้ 0 เพื่อให้ AI ไม่แต่งเรื่องเอง (เน้นความแม่นยำจาก Tools)
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
                   "จงวิเคราะห์คำถามของผู้ใช้อย่างละเอียด และเลือกใช้ปืนให้เหมาะสมกับงบประมาณและสถานการณ์เกราะของศัตรู"
                   "ห้ามเดาหรือแต่งเติมความสามารถของสกิลเองเด็ดขาด หากข้อมูลที่ค้นหามาได้ไม่มีคำอธิบายสกิลนั้นๆ อย่างชัดเจน ให้ตอบผู้ใช้ไปตามตรงว่า 'ไม่พบข้อมูลของสกิลนี้' หรือ 'ข้อมูลสกิลไม่ครบถ้วน'"
                   "ห้ามเดาหรือแต่งเติมข้อมูลเองเด็ดขาด! ถ้าค้นหาชื่อสกิลหรือข้อมูลไม่เจอ ให้ตอบว่า ไม่พบข้อมูลสกิล Curveball เป็นของ Phoenix เสมอ ห้ามนำไปปนกับ Killjoy หรือตัวละครอื่น"
                    "ห้ามเรียกใช้ Tool calculate_economy หรือเดาตัวเลขเงินเอง หากผู้เล่นไม่ได้ระบุจำนวนเงินมาในคำถาม"
                    "เครดิตในเกมสูงสุดคือ 9,000 ห้ามคำนวณเกินนี้"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"), 
    ])

    # สร้าง Agent และ Executor 
    agent = create_tool_calling_agent(llm, tools, prompt)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # verbose=True เพื่อโชว์ Log การทำงานออก Terminal (PoC requirement)
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

    print("=== 🤖 เริ่มต้น Valorant Agentic RAG (Powered by Gemini) ===")
    print("พิมพ์ 'exit' เพื่อออก\n")

    while True:
        user_input = input("👤 คุณ (ผู้เล่น): ")
        if user_input.lower() == 'exit':
            break
        
        # ส่งคำถาม
        response = agent_executor.invoke({"input": user_input})
        
        final_answer = response['output']
        
        # ตรวจสอบและประกอบร่างข้อความในกรณีที่ AI หั่นท่อนมาให้
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
if __name__ == "__main__":
    main()