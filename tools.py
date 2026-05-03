# tools.py
import json
from langchain.tools import tool

# ==========================================
# 1. Mockup Search Tool (รอเชื่อมต่อ Vector DB ภายหลัง)
# ==========================================
@tool
def search_valorant_strategy(query: str) -> str:
    """
    ใช้ Tool นี้เมื่อต้องการค้นหาข้อมูลเชิงกลยุทธ์ เช่น แผนการเล่น, จุดวาง Smoke, หรือ Lineup
    Input: คำที่ต้องการค้นหา (str)
    """
    # TODO: เชื่อมต่อกับ Vector DB ของเพื่อนคนที่ 1
    # ตอนนี้ return ข้อมูลจำลองไปก่อน
    if "smoke" in query.lower() and "bind" in query.lower():
         return "บนด่าน Bind ฝั่งป้องกัน Brimstone ควร Smoke ที่ U-Hall และ Short A"
    return "ไม่พบข้อมูลกลยุทธ์ที่ตรงกับคำค้นหา"

# ==========================================
# 2. Economy Calculator Tool
# ==========================================
@tool
def calculate_economy(current_money: int, last_round_result: str) -> str:
    """
    ใช้ Tool นี้เมื่อผู้ใช้ถามเรื่องการจัดการเงิน หรือควรรอซื้อปืนอะไรในรอบถัดไป
    Input:
    - current_money: จำนวนเงินที่มี (int)
    - last_round_result: 'win' หรือ 'loss' (str)
    """
    # ตัวอย่าง Logic แบบง่าย (คุณสามารถปรับให้ซับซ้อนตามกติกาจริงได้)
    bonus_money = 3000 if last_round_result == "win" else 1900
    expected_money = current_money + bonus_money
    
    if expected_money >= 3900:
        return f"คุณจะมีเงินประมาณ {expected_money} Creds แนะนำให้ Full Buy (ซื้อ Vandal/Phantom + เกราะใหญ่)"
    elif expected_money >= 2000:
         return f"คุณจะมีเงินประมาณ {expected_money} Creds แนะนำให้ Force Buy ปืนกลเบาหรือลูกซอง"
    else:
        return f"คุณจะมีเงินแค่ {expected_money} Creds แนะนำให้ Eco (Save) เพื่อรอเล่นรอบหน้า"

# ==========================================
# 3. Exact Agent Info Tool
# ==========================================
@tool
def get_agent_role(agent_name: str) -> str:
    """
    ใช้ Tool นี้เพื่อหา Role (ตำแหน่ง) ของ Agent ที่ระบุอย่างแม่นยำ
    Input: ชื่อ Agent (str) เช่น 'Jett', 'Omen'
    """
    database = {
        "jett": "Duelist",
        "omen": "Controller",
        "killjoy": "Sentinel",
        "sova": "Initiator"
    }
    role = database.get(agent_name.lower())
    if role:
        return f"{agent_name} เป็นตำแหน่ง {role}"
    return f"ไม่พบข้อมูลของ {agent_name}"