import os
from dotenv import load_dotenv

# 1. นำเข้า Library (ใช้เฉพาะตัวที่จำเป็น)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from tools import search_agent_strategy, search_map_strategy, search_weapon_strategy, search_update_patch, calculate_economy
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder

load_dotenv()

# 2. โหลด API Key ของ Gemini
gemini_api_key = os.environ.get("GOOGLE_API_KEY")

def load_and_tag_data(file_path, category):
    from langchain_core.documents import Document
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # แยกข้อมูลตามหัวข้อ ### (เช่น ### Sage, ### Bandit)
    sections = content.split("### ")
    tagged_docs = []
    
    for section in sections:
        if not section.strip(): continue
        
        # ดึงบรรทัดแรกมาเป็นชื่อ (name) สำหรับทำ Metadata
        lines = section.strip().split("\n")
        item_name = lines[0].strip().lower()
        
        # ติดป้ายกำกับ (Metadata) ให้แต่ละก้อนข้อมูล
        doc = Document(
            page_content="### " + section,
            metadata={"category": category, "name": item_name}
        )
        tagged_docs.append(doc)
    return tagged_docs

def main():
    # 3. ตั้งค่า LLM เป็น Gemini
    llm = ChatGoogleGenerativeAI(
        google_api_key=gemini_api_key,
        model="gemini-2.5-flash", 
        temperature=0, 
        max_tokens=3000
    )

    tools = [
        search_agent_strategy, 
        search_map_strategy, 
        search_weapon_strategy, 
        search_update_patch, 
        calculate_economy
    ]

    # 4. สร้าง Prompt พร้อมช่องใส่ประวัติการคุย (chat_history)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "คุณคือผู้ช่วยโค้ชเกม Valorant สุดอัจฉริยะ หน้าที่ของคุณคือตอบคำถามและให้คำแนะนำผู้เล่น\n"
                   "**กฎสำคัญ:**\n"
                   "1. เรียกใช้ Tool หลายตัวต่อเนื่องกันหากจำเป็น\n"
                   "2. ห้ามแนะนำไอเทมจากเกมอื่น สกิลผูกกับตัวละครเท่านั้น\n"
                   "3. ห้ามเดาคำตอบเอง หากไม่พบข้อมูลให้แจ้งระบบขัดข้อง\n"
                   "4. ทับศัพท์ภาษาอังกฤษเสมอ (เช่น Force Buy, Eco)"
                   "ถ้าผลลัพธ์จาก Tool ไม่มีชื่อ Agent ที่ผู้ใช้ถาม (เช่น Clove)"
                    "ให้ถือว่า ไม่พบข้อมูล ทันที ห้ามใช้ข้อมูลจาก Agent อื่นแทนเด็ดขาด"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"), 
    ])

    # 5. สร้าง Agent และ Executor 
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # ลบส่วน memory เก่าออก และใช้ verbose=True เพื่อโชว์ Log (ตามเกณฑ์อาจารย์)
    agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    max_iterations=10,          # 1. เพิ่มจำนวนครั้งที่ให้ Agent คิด (ค่าเริ่มต้นมักจะเป็น 3-5)
    handle_parsing_errors=True  # 2. ป้องกัน Agent พังเวลา LLM คืนค่าฟอร์แมตแปลกๆ มา
)
    
    # 6. สมุดจดประวัติการคุยแบบ Manual
    chat_history = []
    
    print("=== ✨ เริ่มต้น Valorant Agentic RAG (Powered by Gemini) ===")
    print("พิมพ์ 'exit' เพื่อออก\n")

    while True:
        user_input = input("👤 คุณ (ผู้เล่น): ")
        if user_input.lower() == 'exit':
            break
        
        try:
            # ส่ง input และประวัติการคุยเข้าไป
            response = agent_executor.invoke({
                "input": user_input, 
                "chat_history": chat_history
            })
            
            final_answer = response['output']
            
            # จัดการข้อความที่ได้รับ
            if isinstance(final_answer, list):
                final_answer = "".join([item.get('text', str(item)) if isinstance(item, dict) else str(item) for item in final_answer])

            print(f"\n🧠 Agent ตอบ:\n{final_answer}\n")
            print("-" * 50)
            
            # 7. บันทึกลงสมุดประวัติ (เพื่อให้คุยต่อเนื่องได้)
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=final_answer))
            
        except Exception as e:
            print(f"\n⚠️ เกิดข้อผิดพลาด: {e}")
            print("-" * 50)

if __name__ == "__main__":
    main()