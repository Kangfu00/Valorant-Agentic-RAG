import os
from dotenv import load_dotenv
# 1. เปลี่ยน Library นำเข้าเป็นของ Google GenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from tools import search_valorant_strategy, calculate_economy, get_agent_role

load_dotenv()
# โหลด API Key
# 2. เปลี่ยนชื่อตัวแปรเป็น GOOGLE_API_KEY (ไปสร้างคีย์ฟรีได้ที่ Google AI Studio)
api_key = os.environ.get("GOOGLE_API_KEY")

def main():
    # 3. ตั้งค่า LLM เป็น Gemini (แนะนำ gemini-1.5-flash เพราะเร็วและรองรับ Tool Calling ได้ดีมาก)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0 # ใช้ 0 เพื่อให้ AI ไม่แต่งเรื่องเอง (เน้นความแม่นยำจาก Tools)
    )

    tools = [search_valorant_strategy, calculate_economy, get_agent_role]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "คุณคือผู้ช่วยโค้ชเกม Valorant สุดอัจฉริยะ หน้าที่ของคุณคือตอบคำถามและให้คำแนะนำผู้เล่น "
                   "คุณต้องใช้ Tools ที่มีอยู่ให้เกิดประโยชน์สูงสุด ห้ามเดาข้อมูลตัวเลขหรือข้อมูล Patch เองเด็ดขาด"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"), 
    ])

    # สร้าง Agent และ Executor 
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # verbose=True เพื่อโชว์ Log การทำงานออก Terminal (PoC requirement)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    print("=== 🤖 เริ่มต้น Valorant Agentic RAG (Powered by Gemini) ===")
    print("พิมพ์ 'exit' เพื่อออก\n")

    while True:
        user_input = input("👤 คุณ (ผู้เล่น): ")
        if user_input.lower() == 'exit':
            break
        
        # ส่งคำถาม
        response = agent_executor.invoke({"input": user_input})
        
        print(f"\n🧠 Agent ตอบ: {response['output']}\n")
        print("-" * 50)

if __name__ == "__main__":
    main()