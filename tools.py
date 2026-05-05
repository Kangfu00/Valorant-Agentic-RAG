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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

def get_or_create_db(db_variable, file_name):
    if db_variable is not None:
        return db_variable
        
    # 2. เรียกใช้โมเดลฟรีจาก HuggingFace (รองรับภาษาไทย)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    index_folder = f"{file_name}_faiss"

    # 3. ระบบเซฟและโหลดแบบ Local 
    if os.path.exists(index_folder):
        print(f"กำลังโหลดฐานข้อมูล {file_name} จากเครื่อง (ไวมาก!)...")
        return FAISS.load_local(index_folder, embeddings, allow_dangerous_deserialization=True)

    print(f"กำลังสร้างฐานข้อมูลใหม่จาก {file_name} (ใช้พลังคอมพิวเตอร์ของคุณเอง)...")
    try:
        loader = TextLoader(file_name, encoding="utf-8")
        docs = loader.load()
        
        md_splits = markdown_splitter.split_text(docs[0].page_content)
        final_splits = text_splitter.split_documents(md_splits)
        
        db = FAISS.from_documents(final_splits, embeddings)
        db.save_local(index_folder) # เซฟลงคอมไว้ใช้รอบหน้า
        
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
    global agent_vector_db
    agent_vector_db = get_or_create_db(agent_vector_db, "money.md") # ชี้ไปที่ไฟล์ใหม่ของคุณ
    
    if agent_vector_db is None:
        return "ระบบฐานข้อมูล money ขัดข้อง"
        
    docs = agent_vector_db.similarity_search(query, k=3)
    return "\n\n".join([f"อ้างอิงจากหัวข้อ: {doc.metadata}\nรายละเอียด: {doc.page_content}" for doc in docs])

@tool
def search_agent_strategy(query: str) -> str:
    """ใช้ Tool นี้เมื่อผู้ใช้ถามเกี่ยวกับ 'ตัวละคร (Agent)' ในเกม"""
    global agent_vector_db
    agent_vector_db = get_or_create_db(agent_vector_db, "Agent.md") # ชี้ไปที่ไฟล์ใหม่ของคุณ
    
    if agent_vector_db is None:
        return "ระบบฐานข้อมูล Agent ขัดข้อง"
        
    docs = agent_vector_db.similarity_search(query, k=3)
    return "\n\n".join([f"อ้างอิงจากหัวข้อ: {doc.metadata}\nรายละเอียด: {doc.page_content}" for doc in docs])

@tool
def search_map_strategy(query: str) -> str:
    """ใช้ Tool นี้เมื่อผู้ใช้ถามเกี่ยวกับ 'แผนที่ (Maps)'"""
    global map_vector_db
    map_vector_db = get_or_create_db(map_vector_db, "map.md") 
    
    if map_vector_db is None:
        return "ระบบฐานข้อมูลแผนที่ขัดข้อง"
        
    docs = map_vector_db.similarity_search(query, k=3)
    return "\n\n".join([f"อ้างอิงจากหัวข้อ: {doc.metadata}\nรายละเอียด: {doc.page_content}" for doc in docs])

@tool
def search_weapon_strategy(query: str) -> str:
    """ใช้ Tool นี้เมื่อผู้ใช้ถามเกี่ยวกับ 'อาวุธปืน' หรือ 'การเงิน'"""
    global gun_vector_db
    gun_vector_db = get_or_create_db(gun_vector_db, "gun.md")
    
    if gun_vector_db is None:
         return "ระบบฐานข้อมูลอาวุธขัดข้อง"
         
    docs = gun_vector_db.similarity_search(query, k=3)
    return "\n\n".join([f"อ้างอิงจากหัวข้อ: {doc.metadata}\nรายละเอียด: {doc.page_content}" for doc in docs])

@tool
def search_update_patch(query: str) -> str:
    """ใช้ Tool นี้เมื่อผู้ใช้ถามถึง 'แพตช์อัปเดต' หรือ 'เมตาล่าสุด'"""
    global update_vector_db
    update_vector_db = get_or_create_db(update_vector_db, "update.md")
    
    if update_vector_db is None:
         return "ระบบฐานข้อมูลแพตช์ขัดข้อง"
         
    docs = update_vector_db.similarity_search(query, k=3)
    return "\n\n".join([f"อ้างอิงจากหัวข้อ: {doc.metadata}\nรายละเอียด: {doc.page_content}" for doc in docs])

@tool
def calculate_economy(current_money: int, last_round_result: str):
    """คำนวณเงินในรอบถัดไปตามกฎของ VALORANT"""
    
    # --- จุดที่ 1: การดัก Error (Guardrails) ---
    if not isinstance(current_money, int):
        return "ข้อผิดพลาด: current_money ต้องเป็นตัวเลขจำนวนเต็มเท่านั้น"
    
    if current_money < 0:
        return "ข้อผิดพลาด: เงินปัจจุบันติดลบไม่ได้ โปรดตรวจสอบข้อมูลอีกครั้ง"

    if current_money > 9000: # กฎของเกมเงินสูงสุดคือ 9000
        current_money = 9000

    # --- จุดที่ 2: Logic การคำนวณ ---
    try:
        # สมมติ Logic พื้นฐาน (คุณสามารถปรับตามสูตรจริงของคุณได้)
        base_win = 3000
        base_loss = 1900
        spike_bonus = 300
        
        if last_round_result.lower() == 'win':
            next_round_money = current_money + base_win + spike_bonus
        else:
            next_round_money = current_money + base_loss
            
        # จำกัดเงินไม่ให้เกิน 9000
        final_money = min(next_round_money, 9000)
        
        return f"คุณจะมีเงินในรอบหน้าประมาณ {final_money} เครดิต แนะนำให้วางแผนการซื้อตามงบนี้"

    except Exception as e:
        # --- จุดที่ 3: ดัก Exception ที่ไม่คาดคิด ---
        return f"เกิดข้อผิดพลาดในการคำนวณ: {str(e)}"