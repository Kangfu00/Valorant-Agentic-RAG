# Valorant AI Assistant (Typhoon)

เว็บแอปนี้เป็นผู้ช่วย AI สำหรับเกม Valorant โดยใช้ `app.py` เป็นไฟล์หลักในการทำงาน

## ว่าทำอะไร

`app.py` จะ:
- สร้างเว็บอินเทอร์เฟซด้วย `Streamlit`
- โหลดคีย์จาก `.env`
- เชื่อมต่อ Typhoon AI ผ่าน `langchain_openai`
- สร้าง Agent ที่ใช้ Tools จาก `tools.py`
- รับคำถามจากผู้ใช้และส่งให้ AI วิเคราะห์
- แสดงคำตอบพร้อมรายละเอียดการทำงาน (Thoughts / Debug)

## ติดตั้งอะไรบ้าง

รันคำสั่งนี้ในโฟลเดอร์โปรเจกต์:

```bash
pip install streamlit python-dotenv langchain-openai langchain-classic
```

หากยังไม่ครบ ให้ติดตั้งเพิ่มเติม:

```bash
pip install langchain
```

## ตั้งค่า `.env`

สร้างไฟล์ `.env` ในโฟลเดอร์เดียวกับ `app.py` แล้วเพิ่ม:

```env
TYPHOON_API_KEY=ใส่_API_KEY_ของคุณที่นี่
```

> หากไม่มีคีย์ โปรแกรมจะไม่สามารถรันได้และจะแสดงข้อความเตือน

## วิธีรัน

จากโฟลเดอร์โปรเจกต์ ให้รัน:

```bash
streamlit run app.py
```

แล้วเปิด URL ที่ Streamlit แจ้ง เช่น `http://localhost:8501`

## วิธีใช้งาน

1. พิมพ์คำถามเกี่ยวกับ Valorant ในช่องแชท
2. กดส่ง แล้วรอ AI ตอบ
3. คำตอบจะปรากฏบนหน้าเว็บ
4. หากมีปัญหา สามารถดูข้อความในส่วน `ดูสิ่งที่เอไอคิด`

## ข้อแนะนำ

- `app.py` เป็นไฟล์หลักในการทำงาน
- `tools.py` เป็นที่เก็บฟังก์ชันช่วยเหลือด้านกลยุทธ์ เช่น ค้นหาตัวละคร แผนที่ อาวุธ และคำนวณเศรษฐกิจ
- ต้องมี `TYPHOON_API_KEY` ใน `.env`

## ไฟล์สำคัญ

- `app.py` — เว็บแอปและ Agent controller
- `tools.py` — ฟังก์ชัน Tools ที่เรียกใช้จาก Agent
- `.env` — เก็บค่า API Key

## ตัวอย่าง `.env`

```env
TYPHOON_API_KEY=sk-xxxxxx
```

## สรุป

โปรเจกต์นี้เป็น Valorant AI Assistant ที่ใช้ `app.py` เป็นศูนย์กลาง ทำงานร่วมกับ Typhoon API และ Streamlit เพื่อให้ผู้ใช้ถามคำถามและรับคำแนะนำเกมได้ทันที
