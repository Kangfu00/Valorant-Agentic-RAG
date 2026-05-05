Markdown
# 🎮 Valorant Agentic RAG

โปรเจกต์ Agentic RAG สำหรับให้คำแนะนำเชิงกลยุทธ์ของเกม Valorant
(พัฒนาโดยทีมวิชา AIE223)

---

## 🛠️ การตั้งค่าโปรเจกต์ก่อนเริ่มงาน (สำคัญมาก)

เพื่อให้ทุกคนสามารถรันโปรเจกต์ในเครื่องของตัวเองได้ ให้ทำตามขั้นตอนการตั้งค่า API Key และติดตั้งไลบรารีดังนี้

### 1. ติดตั้งไลบรารีที่จำเป็น
เปิด Terminal ในโฟลเดอร์โปรเจกต์ แล้วรันคำสั่ง:
```bash
pip install langchain langchain-google-genai langchain-openai langchain-classic python-dotenv
```
### 2. การตั้งค่า API Key (ไฟล์ .env)
⚠️ ห้าม Hardcode (พิมพ์) API Key ลงไปในไฟล์โค้ดโดยตรงเด็ดขาด
ระบบจะอ่าน API Key จากไฟล์ .env ซึ่งไฟล์นี้จะถูกตั้งค่าใน .gitignore เพื่อไม่ให้อัปโหลดขึ้น GitHub (ป้องกันรหัสหลุด) ทุกคนต้องสร้างไฟล์นี้ในเครื่องตัวเอง:
1. ในโฟลเดอร์หลักของโปรเจกต์ ให้คลิกขวา -> New File
2. ตั้งชื่อไฟล์ว่า .env (มีจุดข้างหน้าด้วย)
3. ไปขอรับคีย์ Gemini API ฟรีได้ที่: Google AI Studio
4. ก๊อปปี้คีย์มาวางในไฟล์ .env ตามรูปแบบนี้ (ไม่ต้องมีเครื่องหมาย " "):

### 3. เชื่อมต่อ Typhoon API ต่อจาก Gemini
หากต้องการใช้ Typhoon แทนหรือควบคู่กับ Gemini ให้เพิ่มค่าต่อไปนี้ในไฟล์ `.env`:

```Plaintext
TYPHOON_API_KEY=ใส่_API_KEY_ของคุณที่นี่
```

- `main.py` ใช้งาน Gemini โดยอ่านค่า `GOOGLE_API_KEY`
- `main_typhoon.py` ใช้งาน Typhoon โดยอ่านค่า `TYPHOON_API_KEY`
- ไฟล์ทั้งสองสามารถอยู่ในโปรเจกต์เดียวกันได้ หากเก็บคีย์ไว้ใน `.env` พร้อมกัน

ตัวอย่างการรัน:

```bash
python main.py         # รันด้วย Gemini
python main_typhoon.py # รันด้วย Typhoon
```

> หมายเหตุ: `main_typhoon.py` จะเชื่อมต่อ Typhoon ผ่าน `ChatOpenAI` และตั้ง `base_url` เป็น `https://api.opentyphoon.ai/v1` เพื่อชี้ไปยังเซิร์ฟเวอร์ Typhoon

### 4. ถ้าติดตั้งแล้ว แต่ยังรันไม่ผ่าน
- ตรวจสอบว่าไฟล์ `.env` อยู่ในโฟลเดอร์เดียวกับ `main.py`
- ตรวจสอบว่า `TYPHOON_API_KEY` ถูกตั้งค่าเรียบร้อย
- หากยังไม่พบแพ็กเกจ `langchain_openai` ให้รัน `pip install langchain-openai`
### 🌿 กฎการทำงานกับ Git (Git Workflow)
เพื่อป้องกันไม่ให้โค้ดพังหรือตีกันเวลารวมงาน ห้าม Push โค้ดเข้าสาขา main โดยตรงเด็ดขาด! ให้ทุกคนสร้าง Branch ของตัวเองสำหรับทำงานแต่ละส่วน โดยทำตามขั้นตอนดังนี้:
- Step 1: อัปเดตโค้ดล่าสุดเสมอ
ก่อนเริ่มงานใหม่ ให้แน่ใจว่าคุณอยู่ที่สาขา main และดึงโค้ดล่าสุดของเพื่อนๆ มาก่อน

```Bash
git checkout main
git pull origin main
```
- Step 2: สร้างและสลับไปยัง Branch ของตัวเอง
สร้าง Branch ใหม่โดยตั้งชื่องานให้ชัดเจน (เช่น ชื่อคุณ/ชื่อฟีเจอร์)
ตัวอย่างการตั้งชื่อ: feature/kangfu-tools หรือ fix/memory-bug

```Bash
git checkout -b feature/ชื่อของคุณ-ชื่องาน
```
(คำสั่ง -b คือการสร้างสาขาใหม่และย้ายเข้าไปทำงานในสาขานั้นทันที)

- Step 3: ทำงานและ Save (Commit)
เมื่อเขียนโค้ดเสร็จแล้ว ให้ทำการ Save งานลงใน Branch ของคุณ

```Bash
git add .
git commit -m "อธิบายสั้นๆ ว่าอัปเดตอะไรไปบ้าง เช่น Add economy calculator tool"
```

- Step 4: อัปโหลด Branch ของคุณขึ้น GitHub (Push)
อัปโหลดสาขาที่คุณเพิ่งทำเสร็จขึ้นไปยัง Repository ของกลุ่ม

``` Bash
git push origin feature/ชื่อของคุณ-ชื่องาน
```

- Step 5: สร้าง Pull Request (PR)
1. เข้าไปที่หน้าเว็บ GitHub ของโปรเจกต์เรา
2. กดปุ่ม Compare & pull request แถวๆ กล่องสีเหลืองที่เด้งขึ้นมา
3. แจ้งเพื่อนในกลุ่มให้ช่วยรีวิว (Review) โค้ด หากไม่มีปัญหา ค่อยกดปุ่ม Merge pull request เพื่อรวมโค้ดเข้าเส้น main
---

### 🚀 วิธีรันโปรเจกต์
เมื่อตั้งค่าทุกอย่างเสร็จแล้ว สามารถทดสอบรัน Agent ได้เลยด้วยคำสั่ง:

```Bash
python main.py
```