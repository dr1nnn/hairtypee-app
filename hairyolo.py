import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av

# -------------------- PAGE CONFIG & CSS --------------------
def config_page():
    st.set_page_config(page_title="Hairtype Detection", layout="wide")
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        background-color: #800000 !important;
        color: white !important;
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
        opacity: 1 !important;
    }
    section[data-testid="stSidebar"] div[data-selected="true"] {
        background-color: #A52A2A !important;
        border-radius: 5px;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > div:hover {
        background-color: #993333 !important;
        cursor: pointer;
    }
    .sidebar-title {
        color: white !important;
        font-size: 22px;
        font-weight: bold;
        text-align: left;
        margin-top: 70px;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO("hair_yolobest.pt")

# -------------------- VIDEO TRANSFORMER --------------------
class YOLOVideoTransformer(VideoTransformerBase):
    def __init__(self, model, conf, mirror=True):
        self.model = model
        self.conf = conf
        self.mirror = mirror

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.mirror:
            img = cv2.flip(img, 1)
        results = self.model.predict(img, conf=self.conf)
        annotated = results[0].plot()
        return annotated

# -------------------- PAGES --------------------
def render_sidebar():
    st.sidebar.markdown('<div class="sidebar-title">Navigasi</div>', unsafe_allow_html=True)
    return st.sidebar.radio("Pilih Halaman", ["Beranda", "Deteksi", "Informasi Tipe Rambut"])

def render_footer():
    st.markdown("""
    <style>footer {display:none;}</style>
    <hr style="border:none; border-top:1px solid #ccc; margin-top:50px;"/>
    <div style="text-align:center; padding:10px; color:red;">
        <p style="margin:0; font-size:16px;">&copy; 2025 <strong>Hairtype Detection</strong> — Geldrin Reawaruw</p>
        <p style="font-size:15px;">
            <a href="https://github.com/" target="_blank" style="color:red; text-decoration:none;">GitHub</a> |
            <a href="https://www.instagram.com/gldrin_reawaruw/" target="_blank" style="color:red; text-decoration:none;">Instagram</a> |
            <a href="https://www.linkedin.com/in/geldrin-reawaruw-545230222/" target="_blank" style="color:red; text-decoration:none;">LinkedIn</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_beranda():
    st.markdown("<h1 style='text-align:center;'>APLIKASI DETEKSI TIPE RAMBUT MANUSIA</h1>", unsafe_allow_html=True)
    st.image("img/samping.jpg", caption="Contoh deteksi rambut", use_container_width=True)
    st.info("Unggah foto atau gunakan kamera untuk mendeteksi tipe rambutmu dengan kecerdasan buatan.")

def render_deteksi(model):
    st.markdown("<h1 style='text-align:center;'>DETEKSI TIPE RAMBUT</h1>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Upload Gambar", "Kamera Realtime"])

    with tab1:
        conf = st.slider("Confidence (%)", 10, 100, 50)
        uploaded = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            img_np = np.array(image)
            results = model.predict(img_np, conf=conf/100)
            result_img = results[0].plot()
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Gambar Asli", use_container_width=True)
            with col2:
                st.image(result_img, caption="Hasil Deteksi", use_container_width=True)
    
    with tab2:
        conf = st.slider("Confidence Kamera (%)", 10, 100, 50, key="cam_conf")
        mirror = st.checkbox("Mirror View", value=True)
        webrtc_streamer(
            key="realtime-detection",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=lambda: YOLOVideoTransformer(model, conf=conf/100, mirror=mirror),
            media_stream_constraints={"video": True, "audio": False},
        )

def render_info():
    st.markdown("<h1 style='text-align:center;'>INFORMASI TIPE RAMBUT</h1>", unsafe_allow_html=True)
    st.image("img/straight1.png", caption="Contoh rambut lurus", use_container_width=True)
    st.image("img/wavy1.png", caption="Contoh rambut bergelombang", use_container_width=True)
    st.image("img/curly1.png", caption="Contoh rambut keriting", use_container_width=True)
    st.image("img/coily1.png", caption="Contoh rambut sangat keriting", use_container_width=True)

def main():
    config_page()
    menu = render_sidebar()
    model = load_model()

    if menu == "Beranda":
        render_beranda()
    elif menu == "Deteksi":
        render_deteksi(model)
    elif menu == "Informasi Tipe Rambut":
        render_info()

    render_footer()

if __name__ == "__main__":
    main()
