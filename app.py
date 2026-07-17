import cv2
import time
import requests
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# --- 1. ตั้งค่าหน้าเว็บให้สวยงามและรองรับมือถือ (Responsive) ---
st.set_page_config(page_title="AI Scissors Detection", page_icon="✂️", layout="centered")

st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { font-size: 2.2rem !important; text-align: center; }
    p { text-align: center; }
    
    /* ล็อกขนาดกล่องวิดีโอ (ป้องกันหน้าจอกระตุกตอนเปิดกล้องบนมือถือ) */
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
    
    /* ซ่อนเมนูเลือกอุปกรณ์ของ WebRTC ที่ไม่ได้ใช้งานเพื่อความสะอาดตา */
    div[data-testid="stSelectbox"] {
        display: none !important;
    }
    
    /* จัดปุ่ม START / STOP ให้อยู่กึ่งกลาง */
    div.element-container:has(button) {
        display: flex;
        justify-content: center;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("✂️ AI Scissors Detection System")
st.write("ระบบตรวจจับวัตถุอัจฉริยะแบบเรียลไทม์ รองรับการใช้งานผ่านมือถือ")
st.markdown("---")

# --- 2. สร้างส่วนควบคุมการตั้งค่า (Sidebar) ---
st.sidebar.header("⚙️ การตั้งค่าระบบ")
enable_discord = st.sidebar.toggle("เปิดระบบส่งแจ้งเตือน Discord", value=False)

DISCORD_WEBHOOK_URL = st.sidebar.text_input(
    "Discord Webhook URL", 
    value="https://discord.com/api/webhooks/1527568605670281316/c2oHzO7qNxOwpMyVMLCZaSEEcz3HyfALh2XmrQ7xFoWVMMgFd4Jzk8o3vH4hkZH7WJyc",
    type="password"
)

# --- 3. โหลดโมเดลเข้าหน่วยความจำ ---
MODEL_PATH = 'scissors_yolov8_e50/train_result/weights/best.pt'

@st.cache_resource
def load_model():
    try:
        return YOLO(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ ไม่สามารถโหลดไฟล์โมเดลได้ (เช็กตำแหน่งไฟล์ {MODEL_PATH}): {e}")
        return None

model = load_model()

# --- 4. ใช้ Class เป็นตัวกลางส่งผ่านข้อมูลข้าม Thread (แก้ปัญหาภาพค้าง / missing ScriptRunContext) ---
class AppState:
    last_alert_time = 0
    alert_cooldown = 15  # วินาที

def send_discord_alert(webhook_url, message, frame_bgr):
    try:
        payload = { "content": message }
        _, img_encoded = cv2.imencode('.jpg', frame_bgr)
        image_bytes = img_encoded.tobytes()
        
        files = { "file": ("detected.jpg", image_bytes, "image/jpeg") }
        response = requests.post(webhook_url, data=payload, files=files)
        return response.status_code in [200, 204]
    except:
        return False

# --- 5. ฟังก์ชันหลักในการแกะเฟรมจากกล้องเว็บแคมมาประมวลผล ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24") # แปลงเป็นอาเรย์ BGR ของ OpenCV
    
    if model is not None:
        # ยัดโมเดลทำงาน
        results = model(img, stream=True, verbose=False)
        
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
                    
                    # วาดกรอบสี่เหลี่ยมและชื่อวัตถุ
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(img, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 💡 เช็กการแจ้งเตือนโดยใช้ข้อมูลที่ดึงมาจากภายนอก Thread (หลีกเลี่ยง st.session_state)
        current_time = time.time()
        if object_detected and enable_discord:
            if (current_time - AppState.last_alert_time > AppState.alert_cooldown):
                msg = f"🔔 [Streamlit Alert] ตรวจพบ: {detected_name} (Confidence: {conf_score:.2f})"
                # ส่งค่าจากปุ่มหน้าเว็บเข้าไปตรงๆ
                send_discord_alert(DISCORD_WEBHOOK_URL, msg, img)
                AppState.last_alert_time = current_time

    return frame.from_ndarray(img, format="bgr24")

# --- 6. เปิดการเชื่อมต่อระบบกล้องผ่าน WebRTC ---
if model is not None:
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    st.write("### 📸 กดปุ่ม START ด้านล่างเพื่อเริ่มเปิดกล้อง")
    
    webrtc_streamer(
        key="scissors-detection-system",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {"facingMode": "environment"},
            "audio": False
        },
        async_processing=True,
    )

st.info("🔒 ข้อมูลระบบ: ภาพประมวลผลสดบนหน้าเว็บเบราว์เซอร์ และจะส่งออกไปยัง Discord Webhook ของคุณเฉพาะเมื่อตรวจพบวัตถุเท่านั้น")