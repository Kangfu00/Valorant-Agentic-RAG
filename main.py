import os
from dotenv import load_dotenv

# 1. นำเข้า Library (ใช้เฉพาะตัวที่จำเป็น)
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from tools import search_agent_strategy, search_map_strategy, search_weapon_strategy, search_update_patch, calculate_economy
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder

load_dotenv()

# 2. โหลด API Key ของ Typhoon
typhoon_api_key = os.environ.get("TYPHOON_API_KEY")

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

    # 4. สร้าง Prompt พร้อมช่องใส่ประวัติการคุย (chat_history)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "คุณคือผู้ช่วยโค้ชเกม Valorant สุดอัจฉริยะ หน้าที่ของคุณคือตอบคำถามและให้คำแนะนำผู้เล่น\n"
                   "**กฎสำคัญ:**\n"
                   "1. เรียกใช้ Tool หลายตัวต่อเนื่องกันหากจำเป็น\n"
                   "2. ห้ามแนะนำไอเทมจากเกมอื่น สกิลผูกกับตัวละครเท่านั้น\n"
                   "3. ห้ามเดาคำตอบเอง หากไม่พบข้อมูลให้แจ้งระบบขัดข้อง\n"
                   "4. ทับศัพท์ภาษาอังกฤษเสมอ (เช่น Force Buy, Eco)"
                   "ห้ามระบุราคาอาวุธหรือชื่อสกิลที่ไม่ได้ระบุไว้ใน Search Results เด็ดขาด หากไม่ทราบราคาให้ตอบเพียงชื่อปืนและบอกว่าเป็นตัวเลือกที่ประหยัดกว่า"
                   "ห้ามเดาหรือแต่งเติมความสามารถของสกิลเองเด็ดขาด หากข้อมูลที่ค้นหามาได้ไม่มีคำอธิบายสกิลนั้นๆ อย่างชัดเจน ให้ตอบผู้ใช้ไปตามตรงว่า 'ไม่พบข้อมูลของสกิลนี้' หรือ 'ข้อมูลสกิลไม่ครบถ้วน'"
                   "ห้ามเดาหรือแต่งเติมข้อมูลเองเด็ดขาด! ถ้าค้นหาชื่อสกิลหรือข้อมูลไม่เจอ ให้ตอบว่า ไม่พบข้อมูลสกิล Curveball เป็นของ Phoenix เสมอ ห้ามนำไปปนกับ Killjoy หรือตัวละครอื่น"
                    "ห้ามเรียกใช้ Tool calculate_economy หรือเดาตัวเลขเงินเอง หากผู้เล่นไม่ได้ระบุจำนวนเงินมาในคำถาม"
                    "เครดิตในเกมสูงสุดคือ 9,000 ห้ามคำนวณเกินนี้"
                    "หากในข้อมูลที่ค้นหา (Search Results) ไม่ระบุราคาปืนหรือชื่อสกิลที่ชัดเจน ห้ามมโนตัวเลขหรือชื่อขึ้นมาเองเด็ดขาด ให้ตอบกว้างๆ ว่าเป็นปืนราคาประหยัด หรือแจ้งว่าไม่พบข้อมูลราคาที่แน่ชัด"
                    "หากในข้อมูลที่ค้นหา (Search Results) ไม่ระบุราคาที่ชัดเจน ห้ามระบุตัวเลขราคาเองเด็ดขาด ให้แนะนำเพียงชื่อปืนและบอกว่าเป็นตัวเลือกที่ประหยัดกว่าเท่านั้น"
                    "ไอเทมที่ซื้อได้มีเพียง อาวุธ และ เกราะ เท่านั้น ส่วนสกิลต้องเรียกชื่อตามที่ระบุไว้ในข้อมูล Agent เท่านั้น"
                    "ให้ตอบคำถามโดยใช้ข้อมูลจาก Context ที่ได้รับเท่านั้น หากข้อมูลไม่ระบุราคาหรือชื่อสกิล ห้ามคาดเดาเองเด็ดขาด"
                    "5. **กฎเหล็กด้านความถูกต้อง:** ห้ามคาดเดาราคาอาวุธหรือชื่อสกิลเองเด็ดขาด "
                    "หากในข้อมูลที่ค้นหา (Search Results) ไม่ระบุตัวเลขราคาหรือชื่อสกิลที่ชัดเจน "
                    "ให้ตอบว่า 'ข้อมูลส่วนนี้ไม่ปรากฏในฐานข้อมูล' และแนะนำเฉพาะข้อมูลที่มีอยู่จริงเท่านั้น "
                    "จงยึดถือข้อมูลจากไฟล์ .md เป็นความจริงสูงสุด (Ground Truth) มากกว่าความรู้เดิมที่มี"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"), 
    ])

    # 5. สร้าง Agent และ Executor 
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # ลบส่วน memory เก่าออก และใช้ verbose=True เพื่อโชว์ Log (ตามเกณฑ์อาจารย์)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # 6. สมุดจดประวัติการคุยแบบ Manual
    chat_history = []
    
    print("=== 🌪️ เริ่มต้น Valorant Agentic RAG (Powered by Typhoon) ===")
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