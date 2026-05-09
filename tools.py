import os
from langchain.tools import tool
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
# 1. เปลี่ยน Import มาใช้ HuggingFace
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

agent_vector_db = None
map_vector_db = None
gun_vector_db = None
update_vector_db = None
money_vector_db = None

headers_to_split_on = [
    ("#", "Main Topic"),
    ("##", "Sub Topic"),
    ("###", "Specific Item")
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

def get_or_create_db(db_variable, file_name):
    if db_variable is not None:
        return db_variable
        
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    index_folder = f"{file_name}_faiss"

    if os.path.exists(index_folder):
        print(f"กำลังโหลดฐานข้อมูล {file_name} จากเครื่อง (ไวมาก!)...")
        return FAISS.load_local(index_folder, embeddings, allow_dangerous_deserialization=True)

    print(f"กำลังสร้างฐานข้อมูลใหม่จาก {file_name} (ใช้พลังคอมพิวเตอร์ของคุณเอง)...")
    try:
        loader = TextLoader(file_name, encoding="utf-8")
        docs = loader.load()
        
        md_splits = markdown_splitter.split_text(docs[0].page_content)
        
        # --- จุดที่แก้ไข: ระบุหมวดหมู่ให้ตรงกับไฟล์ ---
        category = "ข้อมูล"
        if "gun" in file_name.lower(): category = "อาวุธ"
        elif "agent" in file_name.lower(): category = "เอเจนต์"
        elif "map" in file_name.lower(): category = "แผนที่"
        elif "money" in file_name.lower(): category = "ระบบเศรษฐกิจ"
        
        for doc in md_splits:
            header_text = ""
            if "Main Topic" in doc.metadata:
                header_text += f"หมวดหมู่: {doc.metadata['Main Topic']}\n"
            if "Sub Topic" in doc.metadata:
                header_text += f"หัวข้อ: {doc.metadata['Sub Topic']}\n"
            if "Specific Item" in doc.metadata:
                header_text += f"รายละเอียด: {doc.metadata['Specific Item']}\n"
            
            # นำชื่อหัวข้อทั้งหมดกลับไปแปะหน้าเนื้อหา เพื่อให้ AI ค้นหาคำเจอ
            doc.page_content = header_text + doc.page_content
                
        final_splits = text_splitter.split_documents(md_splits)
        
        db = FAISS.from_documents(final_splits, embeddings)
        db.save_local(index_folder) 
        
        return db
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการโหลด {file_name}: {e}")
        return None

# ==========================================
# Tools สำหรับ Agent
# ==========================================

@tool
def search_agent_money(query: str) -> str:
    """ใช้ Tool นี้เมื่อผู้ใช้ถามเกี่ยวกับการจัดการเงิน (Economy), การซื้อของ (Buy/Save/Force), หรือกลยุทธ์การตัดสินใจตามสถานการณ์ในแต่ละรอบ"""
    global money_vector_db 
    money_vector_db = get_or_create_db(money_vector_db, "money.md") 
    
    if money_vector_db is None:
        return "ระบบฐานข้อมูล money ขัดข้อง"
        
    docs = money_vector_db.similarity_search(query, k=5)
    if not docs: # ถ้าหาไม่เจอเลย
        return "ไม่มีข้อมูลในระบบ กรุณาหยุดค้นหาและแจ้งผู้ใช้ว่าไม่พบข้อมูล"
    return "\n\n".join([f"อ้างอิงจากหัวข้อ: {doc.metadata}\nรายละเอียด: {doc.page_content}" for doc in docs])
    

@tool
def search_agent_strategy(query: str) -> str:
    """ใช้ Tool นี้เมื่อผู้ใช้ถามเกี่ยวกับ Agent เช่น สกิล (abilities), role, หรือวิธีเล่นของตัวละครในเกม Valorant"""
    global agent_vector_db
    agent_vector_db = get_or_create_db(agent_vector_db, "Agent.md") # ชี้ไปที่ไฟล์ใหม่ของคุณ
    
    if agent_vector_db is None:
        return "ระบบฐานข้อมูล Agent ขัดข้อง"
        
    docs = agent_vector_db.similarity_search(query, k=5)
    if not docs: # ถ้าหาไม่เจอเลย
        return "ไม่มีข้อมูลในระบบ กรุณาหยุดค้นหาและแจ้งผู้ใช้ว่าไม่พบข้อมูล"
    return "\n\n".join([f"อ้างอิงจากหัวข้อ: {doc.metadata}\nรายละเอียด: {doc.page_content}" for doc in docs])

@tool
def search_map_strategy(query: str) -> str:
    """ใช้ Tool นี้เมื่อผู้ใช้ถามเกี่ยวกับ 'แผนที่ (Maps)'"""
    global map_vector_db
    map_vector_db = get_or_create_db(map_vector_db, "map.md") 
    
    if map_vector_db is None:
        return "ระบบฐานข้อมูลแผนที่ขัดข้อง"
        
    docs = map_vector_db.similarity_search(query, k=5)
    if not docs: # ถ้าหาไม่เจอเลย
        return "ไม่มีข้อมูลในระบบ กรุณาหยุดค้นหาและแจ้งผู้ใช้ว่าไม่พบข้อมูล"
    return "\n\n".join([f"อ้างอิงจากหัวข้อ: {doc.metadata}\nรายละเอียด: {doc.page_content}" for doc in docs])

@tool
def search_weapon_strategy(query: str) -> str:
    """ใช้ Tool นี้ทุกครั้งเมื่อผู้ใช้ถามเกี่ยวกับข้อมูลอาวุธปืน หรือ เกราะ (Shields/Armor)!
    **ข้อควรระวังสำคัญสำหรับการตั้งคำค้นหา (query):**
    - หากถามถึง 'ราคา' ต้องใส่คำว่า "Price" ลงไปใน query ด้วย เช่น "Price Guardian", "Price Heavy Armor", "Price Light Armor"
    - หากถามถึง 'ดาเมจ' ต้องใส่คำว่า "Damage" ลงไปใน query ด้วย เช่น "Damage Vandal"
    ห้ามใช้ภาษาไทยในการค้นหาสถิติ (ห้ามใช้คำว่า ราคา หรือ ดาเมจ ใน query)"""
    
    global gun_vector_db
    gun_vector_db = get_or_create_db(gun_vector_db, "gun.md")
    
    if gun_vector_db is None:
         return "ระบบฐานข้อมูลอาวุธขัดข้อง"
         
    docs = gun_vector_db.similarity_search(query, k=5)
    if not docs: # ถ้าหาไม่เจอเลย
        return "ไม่มีข้อมูลในระบบ กรุณาหยุดค้นหาและแจ้งผู้ใช้ว่าไม่พบข้อมูล"
    return "\n\n".join([f"อ้างอิงจากหัวข้อ: {doc.metadata}\nรายละเอียด: {doc.page_content}" for doc in docs])

@tool
def search_update_patch(query: str) -> str:
    """ใช้ Tool นี้เมื่อผู้ใช้ถามถึง 'แพตช์อัปเดต' หรือ 'เมตาล่าสุด'"""
    global update_vector_db
    update_vector_db = get_or_create_db(update_vector_db, "update.md")
    
    if update_vector_db is None:
         return "ระบบฐานข้อมูลแพตช์ขัดข้อง"
         
    docs = update_vector_db.similarity_search(query, k=5)
    if not docs: # ถ้าหาไม่เจอเลย
        return "ไม่มีข้อมูลในระบบ กรุณาหยุดค้นหาและแจ้งผู้ใช้ว่าไม่พบข้อมูล"
    return "\n\n".join([f"อ้างอิงจากหัวข้อ: {doc.metadata}\nรายละเอียด: {doc.page_content}" for doc in docs])

@tool
def calculate_economy(current_money: int, last_round_result: str, loss_streak: int = 0, is_save_on_loss: bool = False):
    """
    Use this tool ALWAYS when calculating the player's money for the NEXT ROUND.
    - current_money: จำนวนเงินที่มีในรอบปัจจุบัน (integer)
    - last_round_result: ผลการแข่งขันรอบนี้ 'win' (ชนะ) หรือ 'loss' (แพ้)
    - loss_streak: จำนวนรอบที่แพ้ติดต่อกัน (ใส่ 1, 2, หรือ 3 ขึ้นไป) ถ้าชนะให้ใส่ 0
    - is_save_on_loss: ใส่ True ถ้าทีม 'แพ้' แต่ผู้เล่น 'รอดชีวิต' (เซฟปืน) ไม่งั้นใส่ False
    """
    if not isinstance(current_money, int):
        return "ข้อผิดพลาด: current_money ต้องเป็นตัวเลข"
    if current_money < 0:
        return "ข้อผิดพลาด: เงินปัจจุบันติดลบไม่ได้"

    try:
        if last_round_result.lower() == 'win':
            next_round_money = current_money + 3000
        elif last_round_result.lower() == 'loss':
            # ดักกฎ Save Penalty: รอดตอนแพ้ได้แค่ 1000
            if is_save_on_loss:
                next_round_money = current_money + 1000
            else:
                # คำนวณ Loss Streak (1900 -> 2400 -> 2900)
                if loss_streak <= 1:
                    next_round_money = current_money + 1900
                elif loss_streak == 2:
                    next_round_money = current_money + 2400
                else:
                    next_round_money = current_money + 2900
        else:
            return "ข้อผิดพลาด: last_round_result ต้องเป็น 'win' หรือ 'loss' เท่านั้น"
            
        final_money = min(next_round_money, 9000)
        return f"คุณจะมีเงินในรอบหน้าประมาณ {final_money} เครดิต แนะนำให้วางแผนการซื้อตามงบนี้"

    except Exception as e:
        return f"เกิดข้อผิดพลาดในการคำนวณ: {str(e)}"