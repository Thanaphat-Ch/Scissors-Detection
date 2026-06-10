import os
import cv2
import time
import requests
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from dotenv import load_dotenv
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# --- [COMMENTED OUT] ปิดการโหลดตัวแปรความลับ Telegram เพื่อความปลอดภัยของผู้ใช้งาน ---
# load_dotenv()
# TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
# CHAT_ID = os.getenv("CHAT_ID")

# 1. ตั้งค่าหน้าเว็บให้ Responsive และสวยงาม
st.set_page_config(page_title="Scissors Detection App", page_icon="✂️", layout="centered")

# ตกแต่ง UI เพิ่มเติมด้วย CSS เล็กน้อยเพื่อให้ปุ่มกดและวิดีโอดูดีในมือถือ
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { font-size: 2.2rem !important; text-align: center; }
    p { text-align: center; }
    
    /* 1. สร้างกล่อง Placeholder สีดำล็อคขนาดไว้ (ป้องกันหน้าจอกระตุกตอนเปิดกล้อง) */
    div[data-testid="stVideo"] {
        width: 100% !important;
        max-width: 640px !important;
        aspect-ratio: 4 / 3 !important;
        background-color: #111111 !important;
        border-radius: 12px;
        overflow: hidden;
        margin: 0 auto;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }

    /* 2. ซ่อนกล่อง Selectbox (เมนูเลือกอุปกรณ์/ไมค์) ที่ไม่ได้ใช้งาน */
    div[data-testid="stSelectbox"] {
        display: none !important;
    }
    
    /* 3. จัดให้ปุ่ม START / STOP ของ WebRTC อยู่กึ่งกลางหน้าจออย่างสวยงาม */
    div.element-container:has(button) {
        display: flex;
        justify-content: center;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("✂️ LAB09 - AI Scissors Detection")
st.write("ระบบตรวจจับวัตถุอันตรายอัจฉริยะ (Local Processing Mode)")
st.markdown("---")

# 2. โหลดโมเดล (ใช้ st.cache_resource)
MODEL_PATH = 'scissors_yolov8_e50/train_result/weights/best.pt'

@st.cache_resource
def load_model():
    try:
        return YOLO(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ ไม่สามารถโหลดโมเดลได้: {e}")
        return None

model = load_model()

# def send_telegram(message, image_bytes):
#     url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
#     try:
#         files = {'photo': ('detected.jpg', image_bytes, 'image/jpeg')}
#         data = {'chat_id': CHAT_ID, 'caption': message}
#         response = requests.post(url, files=files, data=data)
#         return response.status_code == 200
#     except Exception as e:
#         return False

if 'last_alert_time' not in st.session_state:
    st.session_state.last_alert_time = 0
alert_cooldown = 15

# 3. ฟังก์ชันหลักในการประมวลผลวิดีโอจาก WebRTC
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24") # ดึงเฟรมภาพมาเป็น Numpy Array
    
    # แปลงสีให้เหมาะกับ YOLO (YOLO ชอบ RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img_rgb, stream=True, verbose=False)

    object_detected = False
    detected_name = ""
    conf_score = 0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]

            if conf > 0.3:
                object_detected = True
                detected_name = class_name
                conf_score = conf

                # วาดกรอบบนรูป
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # current_time = time.time()
    # if object_detected and (current_time - st.session_state.last_alert_time > alert_cooldown):
    #     img_alert_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img_pil = Image.fromarray(img_alert_rgb)
    #     from io import BytesIO
    #     buf = BytesIO()
    #     img_pil.save(buf, format="JPEG")
    #     byte_im = buf.getvalue()
    # 
    #     msg = f"🔔 [Mobile Alert] ตรวจพบ {detected_name}! (Confidence: {conf_score:.2f})"
    #     send_telegram(msg, byte_im)
    #     st.session_state.last_alert_time = current_time

    return frame.from_ndarray(img, format="bgr24")

# 4. ปุ่มกดเปิด/ปิด กล้อง และตั้งค่าให้รองรับกล้องมือถือ (Responsive WebRTC)
if model is not None:
    # ตั้งค่าระบบ Turn Server ฟรี
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    st.write("### 📸 กดปุ่ม START เพื่อเริ่มใช้งานกล้อง")
    
    # ส่วนของกล่องควบคุม WebRTC
    webrtc_streamer(
        key="scissors-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        # บังคับให้มือถือเรียกใช้กล้องหลัง (environment) เพื่อความสะดวก
        media_stream_constraints={
            "video": {"facingMode": "environment"},
            "audio": False
        },
        async_processing=True,
    )
    
    st.info("🔒 ความเป็นส่วนตัว: โปรเจกต์นี้ประมวลผลแบบเรียลไทม์บนหน้าเว็บเท่านั้น ไม่มีการบันทึกภาพหรือส่งข้อมูลใดๆ ไปยังเซิร์ฟเวอร์ภายนอก")