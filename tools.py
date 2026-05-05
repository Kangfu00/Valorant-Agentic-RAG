import os
from langchain.tools import tool
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# 1. ตัวแปร Global สำหรับเก็บฐานข้อมูลแยกแต่ละหมวด
agent_vector_db = None
map_vector_db = None
gun_vector_db = None
update_vector_db = None

# 2. ตั้งค่าตัวหั่นข้อความสำหรับ Markdown โดยเฉพาะ
headers_to_split_on = [
    ("#", "Main Topic"),
    ("##", "Sub Topic"),
    ("###", "Specific Item")
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

# 3. ฟังก์ชันตัวช่วย (Helper Function) สำหรับโหลดไฟล์ .md เพื่อลดโค้ดซ้ำซ้อน
def get_or_create_db(db_variable, file_name):
    if db_variable is not None:
        return db_variable
        
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # ตั้งชื่อแฟ้มสำหรับเซฟฐานข้อมูล (เช่น Agent.md_faiss)
    index_folder = f"{file_name}_faiss"

    # 🛡️ 1. ตรวจสอบว่าเคยเซฟฐานข้อมูลนี้ไว้ในเครื่องหรือยัง?
    if os.path.exists(index_folder):
        print(f"กำลังโหลดฐานข้อมูล {file_name} จากที่เซฟไว้ (ไม่เสียโควต้า API)...")
        # โหลดของเก่ามาใช้เลย
        return FAISS.load_local(index_folder, embeddings, allow_dangerous_deserialization=True)

    # 🚀 2. ถ้ายังไม่เคยทำ ค่อยสร้างใหม่และเซฟเก็บไว้
    print(f"กำลังดึงข้อมูลและสร้างฐานข้อมูลใหม่จาก {file_name} (เสียโควต้า API ครั้งเดียว)...")
    try:
        loader = TextLoader(file_name, encoding="utf-8")
        docs = loader.load()
        
        md_splits = markdown_splitter.split_text(docs[0].page_content)
        final_splits = text_splitter.split_documents(md_splits)
        
        db = FAISS.from_documents(final_splits, embeddings)
        
        # เซฟฐานข้อมูลลงโฟลเดอร์ในเครื่อง!
        db.save_local(index_folder) 
        
        return db
    except Exception as e:
        print(f"❌ ไม่พบไฟล์ หรือเกิดข้อผิดพลาดในการโหลด {file_name}: {e}")
        return None

# ==========================================
# Tools สำหรับ Agent หลัก
# ==========================================

@tool
def search_agent_strategy(query: str) -> str:
    """
    ใช้ Tool นี้เมื่อผู้ใช้ถามเกี่ยวกับ 'ตัวละคร (Agent)' ในเกม
    เช่น สกิล, วิธีเล่น, ตำแหน่งการยืน, หรือหน้าที่ของเอเจนต์ต่างๆ
    """
    global agent_vector_db
    agent_vector_db = get_or_create_db(agent_vector_db, "Agent.md")
    
    # 🛡️ เพิ่มโล่ป้องกันตรงนี้!
    if agent_vector_db is None:
        return "ขออภัยครับ ตอนนี้ระบบฐานข้อมูล Agent กำลังติดลิมิตการโหลด กรุณารอสัก 1 นาทีแล้วถามใหม่นะครับ"
        
    docs = agent_vector_db.similarity_search(query, k=3)
    result = "\n\n".join([f"อ้างอิงจากหัวข้อ: {doc.metadata}\nรายละเอียด: {doc.page_content}" for doc in docs])
    return result

@tool
def search_map_strategy(query: str) -> str:
    """
    ใช้ Tool นี้เมื่อผู้ใช้ถามเกี่ยวกับ 'แผนที่ (Maps)' ในเกม Valorant 
    เช่น วิธีบุกตีไซต์, การยืนตำแหน่งป้องกัน, การคุมโซน, โครงสร้างแผนที่
    """
    global map_vector_db
    map_vector_db = get_or_create_db(map_vector_db, "map.md") 
    
    # 🛡️ เพิ่มโล่ป้องกันตรงนี้!
    if map_vector_db is None:
        return "ขออภัยครับ ตอนนี้ระบบฐานข้อมูลแผนที่กำลังติดลิมิตการโหลด กรุณารอสัก 1 นาทีแล้วถามใหม่นะครับ"
        
    docs = map_vector_db.similarity_search(query, k=3)
    result = "\n\n".join([f"อ้างอิงจากหัวข้อ: {doc.metadata}\nรายละเอียด: {doc.page_content}" for doc in docs])
    return result

@tool
def search_weapon_strategy(query: str) -> str:
    """
    ใช้ Tool นี้เมื่อผู้ใช้ถามเกี่ยวกับ 'อาวุธปืน (Weapons)' หรือ 'การเงิน (Economy)'
    เช่น ปืนไหนดีกว่ากัน, ควรซื้อปืนอะไรในรอบ Eco, สถิติดาเมจของปืนต่างๆ
    """
    global gun_vector_db
    gun_vector_db = get_or_create_db(gun_vector_db, "gun.md")
    
    # 🛡️ เพิ่มโล่ป้องกันตรงนี้!
    if gun_vector_db is None:
         return "ขออภัยครับ ตอนนี้ระบบฐานข้อมูลอาวุธกำลังติดลิมิตการโหลด กรุณารอสัก 1 นาทีแล้วถามใหม่นะครับ"
         
    docs = gun_vector_db.similarity_search(query, k=3)
    result = "\n\n".join([f"อ้างอิงจากหัวข้อ: {doc.metadata}\nรายละเอียด: {doc.page_content}" for doc in docs])
    return result

@tool
def search_update_patch(query: str) -> str:
    """
    ใช้ Tool นี้เมื่อผู้ใช้ถามถึง 'แพตช์อัปเดต (Patch Notes)' หรือ 'เมตาล่าสุด'
    เช่น มีอะไรเปลี่ยนไปบ้างในแพตช์ 12.08, โหมดใหม่คืออะไร
    """
    global update_vector_db
    update_vector_db = get_or_create_db(update_vector_db, "update.md")
    
    # 🛡️ เพิ่มโล่ป้องกันตรงนี้!
    if update_vector_db is None:
         return "ขออภัยครับ ตอนนี้ระบบฐานข้อมูลแพตช์กำลังติดลิมิตการโหลด กรุณารอสัก 1 นาทีแล้วถามใหม่นะครับ"
         
    docs = update_vector_db.similarity_search(query, k=3)
    result = "\n\n".join([f"อ้างอิงจากหัวข้อ: {doc.metadata}\nรายละเอียด: {doc.page_content}" for doc in docs])
    return result

@tool
def calculate_economy(current_money: int, last_round_result: str) -> str:
    """
    ใช้ Tool นี้เพื่อคำนวณแบบคณิตศาสตร์เมื่อผู้ใช้ถามว่าจะมีเงินเท่าไหร่ในรอบถัดไป
    Input:
    - current_money: จำนวนเงินที่มีปัจจุบัน (int)
    - last_round_result: 'win' หรือ 'loss' (str)
    """
    bonus_money = 3000 if last_round_result.lower() == "win" else 1900
    expected_money = current_money + bonus_money
    
    if expected_money >= 3900:
        return f"คุณจะมีเงิน {expected_money} Creds แนะนำให้ Full Buy"
    elif expected_money >= 2000:
         return f"คุณจะมีเงิน {expected_money} Creds แนะนำให้ Force Buy"
    else:
        return f"คุณจะมีเงิน {expected_money} Creds แนะนำให้ Eco (Save)"