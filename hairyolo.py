import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration


# -------------------- PAGE CONFIG & CSS --------------------
def config_page():
    st.set_page_config(page_title="Hairtype Detection", layout="wide")
    st.markdown("""
    <style>
    /* Sidebar background & text */
    section[data-testid="stSidebar"] {
        background-color: #800000 !important;
        color: white !important;
    }

    /* Semua teks & label di sidebar */
    section[data-testid="stSidebar"] * {
        color: white !important;
        font-weight: regular !important;
        opacity: 1 !important;
    }

    /* Opsi radio yang dipilih */
    section[data-testid="stSidebar"] div[data-selected="true"] {
        background-color: #A52A2A !important;
        border-radius: 5px;
    }

    /* Hover efek */
    section[data-testid="stSidebar"] div[role="radiogroup"] > div:hover {
        background-color: #993333 !important;
        cursor: pointer;
    }

    /* Teks "Navigasi" */
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


# -------------------- MODEL LOADING --------------------
@st.cache_resource
def load_model():
    return YOLO("hair_yolobest.pt")

# -------------------- HAIRCARE RECOMMENDATION --------------------
def get_haircare_info(label):
    info = {
        "straight": {
            "deskripsi": "Tipe rambut lurus adalah tipe rambut yang jatuh lembut dari akar hingga ujung dengan kilau alami karena minyak mudah tersebar. Namun, jenis ini mudah lepek, kurang bervolume, dan sulit mempertahankan gaya bergelombang atau keriting..",
            "perawatan": "Gunakan sampo ringan & hindari produk berat.",
        },
        "wavy": {
            "deskripsi": "Rambut bergelombang memiliki bentuk “S” yang muncul di bagian tengah hingga ujung rambut, dan cenderung memiliki volume alami lebih banyak dari rambut lurus. Tantangannya adalah mudah kusut, rentan mengembang (frizzy), serta gelombangnya bisa tidak konsisten.",
            "perawatan": "Gunakan sampo bebas sulfat & kondisioner lembap.",
        },
        "curly": {
            "deskripsi": "Rambut ikal memiliki pola keriting yang terlihat jelas, terutama saat kering. Saat basah, rambut bisa tampak lebih lurus namun akan kembali ikal saat mengering. Jenis rambut ini cenderung mudah mengembang, kering, patah, dan susah diatur.",
            "perawatan": "Gunakan 'squish to condish' & handuk microfiber.",
        },
        "coily": {
            "deskripsi": " Rambut ini memiliki pola keriting sangat rapat, berbentuk spiral kecil atau zigzag, dengan tekstur mulai dari kasar hingga sangat kasar. Meskipun terlihat tebal, rambut ini sangat rapuh, mudah kusut, dan rentan rusak jika terlalu sering disisir atau terkena panas berlebih.",
            "perawatan": "Lakukan deep conditioning mingguan, metode LOC.",
        }
    }
    return info.get(label.lower(), {
        "deskripsi": "Informasi tidak tersedia.",
        "perawatan": "Informasi tidak tersedia.",
        "styling": "Informasi tidak tersedia."
    })


# -------------------- UI COMPONENTS --------------------
def render_sidebar():
    st.sidebar.markdown('<div class="sidebar-title">Navigasi</div>', unsafe_allow_html=True)
    return st.sidebar.radio("Pilih Halaman", ["Beranda", "Deteksi", "Informasi Tipe Rambut"])

def render_footer():
    st.markdown("""
    <style>
    footer {
        display: none;
    }
    .reportview-container .main footer, .stApp {
        padding-bottom: 0px;
        margin-bottom: 0px;
    }
    .block-container {
        padding-bottom: 0px !important;
    }
    </style>
    
    <hr style="border: none; border-top: 1px solid #ccc; margin-top: 50px;"/>
    <div style="text-align: center; padding:10px 0 5px 0; color: red;">
        <p style="margin: 0; font-size: 16px;">&copy; 2025 <strong>Hairtype Detection</strong> — Geldrin Reawaruw</p>
        <p style="margin: 5px 0; font-size: 15px;">
            <a href="https://github.com/dr1nnn/" target="_blank" style="color: red; text-decoration: none; margin: 0 10px;">GitHub</a> |
            <a href="https://www.instagram.com/gldrin_reawaruw/" target="_blank" style="color: red; text-decoration: none; margin: 0 10px;">Instagram</a> |
            <a href="https://www.linkedin.com/in/geldrin-reawaruw-545230222/" target="_blank" style="color: red; text-decoration: none; margin: 0 10px;">LinkedIn</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

# -------------------- PAGE: BERANDA --------------------
def render_beranda():
    st.markdown("<h1 style='text-align:center;'>APLIKASI DETEKSI TIPE RAMBUT MANUSIA</h1>", unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)

    col_text, col_img = st.columns([2, 0.9])
    with col_text:
        st.markdown("""
                    <div style='font-size:22px; line-height:1.6; text-align:justify;'>
                    Aplikasi ini adalah alat berbasis kecerdasan buatan (AI) yang membantu kamu mengetahui jenis rambutmu—lurus, bergelombang, keriting, atau sangat keriting—hanya dengan mengunggah foto. Sistem akan menganalisis bentuk dan tekstur rambutmu secara otomatis, lalu menampilkan hasilnya dalam hitungan detik.  
                    <br><br>
                    Mengetahui jenis rambut sangat penting karena setiap tipe rambut membutuhkan perawatan yang berbeda. Dengan aplikasi ini, kamu tidak hanya bisa mengenali jenis rambutmu, tapi juga mendapatkan rekomendasi produk dan cara perawatan yang paling sesuai.
                    </div>
                    """, unsafe_allow_html=True)

    with col_img:
        st.image("img/samping.jpg", caption="Contoh deteksi rambut", use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style='max-width:500px; margin:auto;'>
            <div style='background-color:#800000; padding:20px; border-radius:15px; box-shadow: 2px 2px 6px #444; color:white;'>
                <h4 style='text-align:center;'>Manfaat</h4>
                <ul style='padding-left:22px; font-size:18px; line-height:1; text-align:justify;'>
                    <li>Mengenali tipe rambut secara otomatis</li>
                    <li>Mendapatkan tips perawatan yang sesuai</li>
                    <li>Rekomendasi produk dan tutorial styling</li>
                    <li>Menghemat waktu uji coba produk rambut</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='max-width:500px; margin:auto;'>
            <div style='background-color:#800000; padding:20px; border-radius:15px; box-shadow: 2px 2px 6px #444; color:white;'>
                <h4 style='text-align:center;'>Cara Kerja</h4>
                <ul style='padding-left:22px; font-size:18px; line-height:1; text-align:justify;'>
                    <li>Unggah foto atau buka kamera</li>
                    <li>AI mendeteksi tipe rambut dari gambar</li>
                    <li>Aplikasi menampilkan hasil dan saran perawatan</li>
                    <li>Tersedia video tutorial sesuai tipe rambut</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

# -------------------- PAGE: DETEKSI --------------------
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_model()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model.predict(img, conf=0.5)  # Sesuaikan confidence jika perlu
        plotted_img = results[0].plot()  # Sudah dalam format BGR
        return av.VideoFrame.from_ndarray(plotted_img, format="bgr24")

def render_deteksi(model):
    st.markdown("<h1 style='text-align:center;'>DETEKSI TIPE RAMBUT MANUSIA</h1>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Upload Gambar", "Kamera"])

    with tab1:
        conf = st.slider("Confidence (%)", 10, 100, 50)
        uploaded = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            img_np = np.array(image)
            results = model.predict(img_np, conf=conf / 100)
            result_img = results[0].plot()
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Gambar Asli", use_container_width=True)
            with col2:
                st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

    with tab2:
        conf = st.slider("Confidence Kamera (%)", 30, 100, 50)

        webrtc_streamer(
            key="example",
            video_processor_factory=VideoProcessor,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            media_stream_constraints={"video": True, "audio": False}
        )
        
# -------------------- PAGE: INFORMASI --------------------
def render_info():
    st.markdown("<h1 style='text-align:center;'>INFORMASI TIPE RAMBUT</h1>", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:justify; font-size:22px; line-height:1.6;'>
    Rambut manusia memiliki berbagai tipe yang unik dan dipengaruhi oleh faktor genetik, etnis, serta lingkungan.
    Memahami tipe rambut sangat penting untuk menentukan perawatan yang tepat serta untuk pengembangan produk kecantikan atau medis yang sesuai.
    </div><br>
    """, unsafe_allow_html=True)

    def hair_type_box(title, style_name, image_path, index, description):
        col1, col2 = st.columns([0.5, 1]) if index % 2 == 0 else st.columns([1, 0.5])

        if index % 2 == 0:
            # Gambar kiri, teks kanan
            with col1:
                st.image(image_path, width=500)
            with col2:
                st.markdown(f"""
                    <div style='background-color:#800000; padding:25px; border-radius:15px; 
                                box-shadow: 2px 2px 6px #444; color:white; margin-bottom:30px; text-align:justify;'>
                        <h4>{title} <i>({style_name})</i></h4>
                        <p style='font-size:18px; line-height:1;'>{description}</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            # Teks kiri, gambar kanan
            with col1:
                st.markdown(f"""
                    <div style='background-color:#800000; padding:25px; border-radius:15px; 
                                box-shadow: 2px 2px 6px #444; color:white; margin-bottom:30px; text-align:justify;'>
                        <h4>{title} <i>({style_name})</i></h4>
                        <p style='font-size:18px; line-height:1;'>{description}</p>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.image(image_path, width=500)

    hair_type_box(
        "Tipe Rambut Lurus",
        "Straight",
        "img/straight1.png",
        0,
        """
        Rambut lurus memiliki helai yang jatuh lembut dari akar hingga ujung, dengan kilau alami karena minyak kulit kepala mudah menyebar. 
        Namun, jenis rambut ini cenderung mudah lepek, kurang bervolume, dan sulit mempertahankan gaya rambut bergelombang atau keriting.

        <strong>Kekurangan:
        - Bisa tampak lepek dan kurang bervolume.
        - Rentan terhadap polusi dan cepat terlihat kusam jika tidak dirawat dengan baik.

        <strong>Perawatan:</strong><br>
        - Gunakan sampo yang ringan dan tidak membuat rambut lepek.
        - Gunakan <em>Dove 1 Minute Super Conditioner Hair Fall Rescue</em> untuk membantu mengurangi rambut rontok.
        - Lakukan perawatan mingguan dengan <em>Dove Creambath Hair Growth Ritual</em> untuk menjaga kekuatan akar rambut.

        <strong>Styling:</strong>
        - Gunakan dry shampoo di akar untuk menambah volume.
        - Alat catok bergelombang, rol panas, atau sea salt spray bisa membantu menciptakan tekstur.
        - Gunakan mousse ringan untuk memberikan efek bervolume yang tahan lama.
        """
    )


    hair_type_box(
        "Tipe Rambut Bergelombang",
        "Wavy",
        "img/wavy1.png",
        1,
        """
        Rambut bergelombang memiliki bentuk “S” yang muncul di bagian tengah hingga ujung rambut, dan cenderung memiliki volume alami lebih banyak 
        dari rambut lurus. Tantangannya adalah mudah kusut, rentan mengembang (frizzy), serta gelombangnya bisa tidak konsisten.

        <strong>Perawatan:</strong>
        - Gunakan produk dengan formula pelembap ringan.
        - Setelah keramas, gunakan <em>Dove 1 Minute Super Conditioner Intensive Damage Treatment</em> untuk menghaluskan dan melembapkan rambut.
        - Lakukan creambath seminggu sekali dengan <em>Dove Creambath Hair Growth Ritual</em> untuk nutrisi dan mengunci kelembapan.

        <strong>Styling:</strong>
        - Gunakan metode scrunching atau plopping saat rambut setengah kering. 
        - Keringkan dengan diffuser agar gelombang tetap terbentuk alami. 
        - Tambahkan sea salt spray atau mousse ringan untuk efek bergelombang yang tahan lama.
        """
    )


    hair_type_box(
        "Tipe Rambut Keriting",
        "Curly",
        "img/curly1.png",
        2,
        """
        Rambut ikal memiliki pola keriting yang terlihat jelas, terutama saat kering. Saat basah, 
        rambut bisa tampak lebih lurus namun akan kembali ikal saat mengering. 
        Jenis rambut ini cenderung mudah mengembang, kering, patah, dan susah diatur.

        <strong>Perawatan:</strong>
        - Gunakan sampo yang mengandung argan oil dan vitamin E.
        - Gunakan kondisioner secara rutin untuk menjaga kelembapan lekukan rambut.
        - Aplikasikan kondisioner tanpa bilas setelah keramas.
        - Hindari produk dengan silikon dan asam sulfat.
        - Hindari menyisir dan menguncir rambut terlalu sering.

        <strong>Styling:</strong>
        - Terapkan teknik rake and shake atau finger coiling dengan leave-in conditioner dan curl cream saat rambut setengah basah. 
        - Gunakan diffuser pada suhu rendah untuk mempertahankan bentuk ikal. 
        - Styling gel bisa membantu mempertahankan definisi lebih lama.
        """
    )

    hair_type_box(
        "Tipe Rambut Sangat Keriting",
        "Coily",
        "img/coily1.png",
        3,
        """
        Rambut ini memiliki pola keriting sangat rapat, berbentuk spiral kecil atau zigzag, dengan tekstur mulai dari kasar hingga sangat kasar. 
        Meskipun terlihat tebal, rambut ini sangat rapuh, mudah kusut, dan rentan rusak jika terlalu sering disisir atau terkena panas berlebih.
        
        <strong>Kekurangan:</strong>
        - Rentan terhadap kerusakan akibat panas dan zat kimia.
        - Mudah kusut dan patah bila tidak dirawat dengan hati-hati.

        <strong>Perawatan:</strong>
        - Gunakan kondisioner tanpa bilas dan masker rambut <em>deep conditioning</em>.
        - Gunakan sampo dan kondisioner yang memperbaiki kerusakan.
        - Hindari menyisir terlalu sering dan gunakan produk yang menutrisi dari akar hingga ujung rambut.

        <strong>Styling:</strong>
        - Terapkan gaya pelindung seperti twists, bantu knots, atau box braids untuk menjaga kelembapan dan mengurangi kerusakan. 
        - Teknik twist out atau braid out juga cocok untuk tampilan alami. 
        - Gunakan jari atau sisir bergigi jarang saat menata rambut agar tekstur tidak rusak.
        """
    )

# -------------------- MAIN --------------------
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
