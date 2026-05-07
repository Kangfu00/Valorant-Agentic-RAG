import os
from dotenv import load_dotenv

# 1. นำเข้า Library (แก้ไขชื่อให้ถูกต้อง)
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from tools import search_agent_strategy, search_map_strategy, search_weapon_strategy, search_update_patch, calculate_economy
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder

load_dotenv()

# 2. โหลด API Key ของ Typhoon
typhoon_api_key = os.environ.get("TYPHOON_API_KEY")

def load_and_tag_data(file_path, category):
    from langchain_core.documents import Document
    import re
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # แยกด้วย "## " เพื่อแบ่งหัวข้อ
    sections = re.split(r'\n## ', content)
    tagged_docs = []
    
    for section in sections:
        if not section.strip(): 
            continue
        
        lines = section.strip().split("\n")
        header = lines[0].strip().lower()
        
        # ทำความสะอาดชื่อเพื่อใช้เป็น Metadata
        clean_name = header.replace("weapon:", "").replace("scenario:", "").strip()
        
        doc = Document(
            page_content="## " + section,
            metadata={
                "category": category, 
                "name": clean_name,
                "type": "weapon" if "weapon:" in header else "strategy"
            }
        )
        tagged_docs.append(doc)
    return tagged_docs

def main():
    # 3. ตั้งค่า LLM เป็น Typhoon
    llm = ChatOpenAI(
        api_key=typhoon_api_key,
        base_url="https://api.opentyphoon.ai/v1", 
        model="typhoon-v2.5-30b-a3b-instruct", 
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

    # --- System Prompt (เพิ่มกฎป้องกันความจำเก่าตีกับความจำใหม่) ---
    system_instruction = """คุณคือผู้ช่วยโค้ชเกม Valorant สุดอัจฉริยะ หน้าที่ของคุณคือตอบคำถามและให้คำแนะนำผู้เล่น

**กฎเหล็กที่คุณต้องปฏิบัติตามอย่างเคร่งครัด:**
1. ห้ามตอบคำถามเกี่ยวกับ ราคา (Price), ดาเมจ (Damage), สถิติอาวุธ, หรือสกิลของเอเจนต์ จากความจำของคุณเองเด็ดขาด!
2. เมื่อผู้ใช้ถามถึงอาวุธ (เช่น Guardian, Ghost, Vandal) หรือราคา คุณ **ต้อง** เรียกใช้ Tool `search_weapon_strategy` เสมอ ห้ามข้ามขั้นตอนนี้
3. หากค้นหาผ่าน Tool แล้วไม่พบข้อมูล หรือในรายละเอียดไม่ได้ระบุ 'ตัวเลขราคา' อย่างชัดเจน ให้ตอบตามตรงว่า "ไม่มีข้อมูลในระบบ" ห้ามเดา และห้ามแต่งตัวเลขขึ้นมาเอง
4. หน่วยเงินในเกมคือ "Credits" (เครดิต) เท่านั้น ห้ามใช้ดอลลาร์
5. ถ้าไม่พบข้อมูล Agent (เช่น Clove) ให้ถือว่า "ไม่พบข้อมูล" ทันที ห้ามใช้ข้อมูลตัวอื่นแทน
6. ข้อควรระวัง: เมื่อผู้ใช้ถามคำถามใหม่ คุณต้องเริ่มต้นค้นหาผ่าน Tool ใหม่ทุกครั้ง ห้ามใช้ข้อมูลราคาหรือสถิติเก่าที่เคยตอบไปแล้วมาตอบซ้ำในบริบทใหม่"""

    # 4. สร้าง Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"), 
    ])
    
    # 5. สร้าง Agent และ Executor
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        max_iterations=25,
        handle_parsing_errors=True
    )
    
    # 6. สมุดจดประวัติการคุยแบบ Manual
    chat_history = []
    
    print("=== 🌪️ เริ่มต้น Valorant Agentic RAG (Powered by Typhoon) ===")
    print("พิมพ์ 'exit' เพื่อออก\n")

    while True:
        user_input = input("👤 คุณ (ผู้เล่น): ")
        if user_input.lower() == 'exit':
            break
        
        try:
            # --- แก้ปัญหาถามรอบที่ 4 แล้วเพี้ยน: จำกัดประวัติเหลือแค่ 4 ข้อความล่าสุด ---
            recent_history = chat_history[-4:] if len(chat_history) > 4 else chat_history

            response = agent_executor.invoke({
                "input": user_input, 
                "chat_history": recent_history
            })
            
            final_answer = response['output']
            
            # จัดการ Format ข้อความ
            if isinstance(final_answer, list):
                final_answer = "".join([item.get('text', str(item)) if isinstance(item, dict) else str(item) for item in final_answer])

            print(f"\n🧠 Agent ตอบ:\n{final_answer}\n")
            print("-" * 50)
            
            # 7. บันทึกลงสมุดประวัติ
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=final_answer))
            
        except Exception as e:
            print(f"\n⚠️ เกิดข้อผิดพลาด: {e}")
            print("-" * 50)

if __name__ == "__main__":
    main()