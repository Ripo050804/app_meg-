"""
APLIKASI KLASIFIKASI BATU MEGALITIKUM
Fitur: Kamera, Upload, Info Kelas, Download PDF, Mobile-friendly
Dengan auto-download model dari Google Drive
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageStat
import json
import os
import time
from fpdf import FPDF
import tempfile
from io import BytesIO
import gdown

# ==============================================
# KONFIGURASI HALAMAN
# ==============================================
st.set_page_config(
    page_title="Klasifikasi Batu Megalitikum",
    page_icon="🗿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================================
# DESKRIPSI KELAS
# ==============================================
DESKRIPSI_KELAS = {
    "Arca": "Arca adalah patung yang melambangkan nenek moyang atau dewa. Biasanya berbentuk manusia atau hewan, dan ditemukan di situs megalitik sebagai objek pemujaan.",
    "dolmen": "Dolmen adalah meja batu yang terdiri dari beberapa batu tegak yang menopang batu datar di atasnya. Digunakan sebagai tempat meletakkan sesaji atau untuk upacara.",
    "menhir": "Menhir adalah tugu batu tegak yang didirikan sebagai tanda peringatan atau simbol kekuatan. Biasanya ditemukan berdiri sendiri atau berkelompok.",
    "dakon": "Dakon adalah batu berlubang-lubang yang menyerupai papan permainan congkak. Diduga digunakan untuk ritual atau permainan tradisional.",
    "batu_datar": "Batu datar adalah batu besar berbentuk lempengan yang mungkin digunakan sebagai alas atau tempat duduk dalam upacara adat.",
    "Kubur_batu": "Kubur batu adalah peti mati yang terbuat dari batu, digunakan untuk mengubur mayat pada masa megalitik. Biasanya ditemukan di dalam tanah.",
    "monolit": "Monolit adalah batu tunggal berukuran besar yang didirikan sebagai monumen atau tanda suatu peristiwa penting."
}

# ==============================================
# KONFIGURASI GOOGLE DRIVE
# ==============================================
# FILE_ID dari link Google Drive Anda
FILE_ID = "1anqwxu65GSw2iQr9ISdHUBgh9D3H2uGt"  # <-- SUDAH DIISI

# ==============================================
# FUNGSI DOWNLOAD MODEL DARI GOOGLE DRIVE
# ==============================================
@st.cache_resource
def download_and_load_model():
    """Download model TFLite dari Google Drive jika belum ada"""
    
    model_path = "megalitikum_model.tflite"
    
    # Cek apakah model sudah ada
    if not os.path.exists(model_path):
        with st.spinner("🔄 Mendownload model (96 MB) dari Google Drive..."):
            try:
                url = f"https://drive.google.com/uc?id={FILE_ID}"
                gdown.download(url, model_path, quiet=False)
                st.success("✅ Model berhasil didownload!")
            except Exception as e:
                st.error(f"❌ Gagal download model: {str(e)}")
                st.info("Coba upload model manual atau periksa FILE_ID")
                return None, None, None
    
    # Load model TFLite
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return interpreter, input_details, output_details
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {str(e)}")
        return None, None, None

# ==============================================
# FUNGSI ENHANCEMENT GAMBAR DENGAN PILLOW
# ==============================================
def enhance_image(image):
    """Tingkatkan kualitas gambar untuk prediksi lebih akurat"""
    
    # Konversi ke RGB jika perlu
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 1. Sharpening untuk mengurangi blur
    image = image.filter(ImageFilter.SHARPEN)
    image = image.filter(ImageFilter.SHARPEN)  # Double sharpen untuk blur parah
    
    # 2. Tingkatkan kontras
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.3)
    
    # 3. Tingkatkan ketajaman
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.5)
    
    # 4. Normalisasi brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.1)
    
    return image

def adaptive_enhancement(image, blur_score, brightness, contrast):
    """Enhancement adaptif berdasarkan skor kualitas"""
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Jika blur, tambah sharpening lebih kuat (gunakan nilai blur_score)
    if blur_score < 200:
        for _ in range(3):
            image = image.filter(ImageFilter.SHARPEN)
    
    # Jika kontras rendah, tingkatkan kontras
    if contrast < 40:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
    
    # Jika terlalu gelap, tingkatkan brightness
    if brightness < 80:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.3)
    
    # Jika terlalu terang, kurangi brightness
    elif brightness > 180:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(0.8)
    
    return image

# ==============================================
# FUNGSI DETEKSI OBJEK NON-BATU (TANPA OPENCV)
# ==============================================
def detect_non_megalith(image):
    """
    Deteksi sederhana apakah gambar mengandung objek non-batu
    Menggunakan analisis warna dengan Pillow
    """
    try:
        # Konversi ke RGB
        img = image.convert('RGB')
        
        # Dapatkan statistik warna
        r, g, b = img.split()
        r_mean = np.mean(np.array(r))
        g_mean = np.mean(np.array(g))
        b_mean = np.mean(np.array(b))
        
        # Batu cenderung memiliki nilai RGB yang mirip (grayscale)
        rgb_variance = np.var([r_mean, g_mean, b_mean])
        
        # Deteksi warna hijau (daun, rumput)
        green_dominance = (g_mean > r_mean * 1.2) and (g_mean > b_mean * 1.2)
        
        # Deteksi warna biru (langit, air)
        blue_dominance = (b_mean > r_mean * 1.3) and (b_mean > g_mean * 1.3)
        
        # Deteksi warna cerah
        color_variance = np.var([r_mean, g_mean, b_mean])
        high_color_variance = color_variance > 500
        
        # Logika deteksi
        if green_dominance:
            return True, "Gambar didominasi warna hijau (mungkin tumbuhan/pohon)"
        elif blue_dominance:
            return True, "Gambar didominasi warna biru (mungkin langit/air)"
        elif high_color_variance and rgb_variance > 200:
            return True, "Gambar memiliki variasi warna tinggi (mungkin bukan batu)"
        elif rgb_variance > 300:
            return True, "Warna tidak seragam (mungkin objek non-batu)"
        else:
            return False, "Objek terdeteksi sebagai potensi batu"
            
    except Exception as e:
        return False, f"Error deteksi: {str(e)}"

# ==============================================
# FUNGSI DETEKSI KUALITAS GAMBAR (TANPA OPENCV)
# ==============================================
def cek_kualitas_gambar(image):
    """Deteksi kualitas gambar (buruk/sedang/baik) menggunakan Pillow"""
    try:
        # Konversi ke grayscale untuk analisis
        gray = image.convert('L')
        img_array = np.array(gray)
        
        # Hitung brightness rata-rata
        brightness = np.mean(img_array)
        
        # Hitung kontras (standar deviasi)
        contrast = np.std(img_array)
        
        # Estimasi blur menggunakan Laplacian variance sederhana
        # Gunakan filter gradient sederhana
        from scipy import ndimage
        laplacian = ndimage.laplace(img_array)
        laplacian_var = np.var(laplacian)
        
        # Klasifikasi kualitas
        kualitas = "Baik"
        pesan = []
        rekomendasi = []
        
        # Deteksi blur
        if laplacian_var < 100:
            kualitas = "Buruk"
            pesan.append("• Gambar terlalu blur/kabur")
            rekomendasi.append("Gambar akan di-sharpen otomatis")
        elif laplacian_var < 200:
            if kualitas == "Baik":
                kualitas = "Sedang"
            pesan.append("• Gambar sedikit blur")
            rekomendasi.append("Aplikasi akan meningkatkan ketajaman")
        
        # Deteksi brightness
        if brightness < 50:
            if kualitas == "Baik":
                kualitas = "Sedang"
            pesan.append("• Gambar terlalu gelap")
            rekomendasi.append("Kecerahan akan ditingkatkan")
        elif brightness > 200:
            if kualitas == "Baik":
                kualitas = "Sedang"
            pesan.append("• Gambar terlalu terang")
            rekomendasi.append("Kecerahan akan dikurangi")
        
        # Deteksi kontras rendah
        if contrast < 40:
            if kualitas == "Baik":
                kualitas = "Sedang"
            pesan.append("• Kontras gambar rendah")
            rekomendasi.append("Kontras akan ditingkatkan")
        
        # Gabungkan pesan
        pesan_text = "\n".join(pesan) if pesan else "Kualitas gambar baik"
        rekomendasi_text = "\n".join(rekomendasi) if rekomendasi else "Tidak perlu enhancement"
        
        return kualitas, pesan_text, rekomendasi_text, laplacian_var, brightness, contrast
        
    except Exception as e:
        return "Tidak diketahui", f"Error analisis: {str(e)}", "", 0, 0, 0

# ==============================================
# FUNGSI LOAD CLASS NAMES
# ==============================================
@st.cache_data
def load_class_names():
    try:
        with open('class_names.json', 'r') as f:
            return json.load(f)
    except:
        return list(DESKRIPSI_KELAS.keys())

@st.cache_data
def load_model_info():
    try:
        with open('model_info.json', 'r') as f:
            return json.load(f)
    except:
        return {'test_accuracy': 0.9782, 'best_val_accuracy_phase2': 0.9880}

# ==============================================
# FUNGSI PREDIKSI TFLITE
# ==============================================
def predict_tflite(interpreter, input_details, output_details, image):
    img = image.convert('RGB').resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    input_data = np.expand_dims(img_array, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

# ==============================================
# FUNGSI BUAT PDF HASIL
# ==============================================
def buat_pdf_hasil(nama_file, kelas, confidence, top3, deskripsi, kualitas="", warning=""):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16, style='B')
    pdf.cell(200, 10, txt="LAPORAN KLASIFIKASI BATU MEGALITIKUM", ln=1, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"File: {nama_file}", ln=1)
    pdf.cell(200, 10, txt=f"Hasil Prediksi: {kelas}", ln=1)
    pdf.cell(200, 10, txt=f"Confidence: {confidence:.2%}", ln=1)
    if kualitas:
        pdf.cell(200, 10, txt=f"Kualitas Gambar: {kualitas}", ln=1)
    if warning:
        pdf.cell(200, 10, txt=f"Catatan: {warning}", ln=1)
    pdf.ln(5)
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, txt="Top 3 Prediksi:", ln=1)
    pdf.set_font("Arial", size=12)
    for i, (k, c) in enumerate(top3, 1):
        pdf.cell(200, 10, txt=f"  {i}. {k}: {c:.2%}", ln=1)
    pdf.ln(5)
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, txt="Deskripsi:", ln=1)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=deskripsi)
    return pdf.output(dest='S').encode('latin1')

# ==============================================
# LOAD SEMUA DATA
# ==============================================
# Download dan load model dari Google Drive
interpreter, input_details, output_details = download_and_load_model()
class_names = load_class_names()
model_info = load_model_info()

# ==============================================
# SIDEBAR
# ==============================================
with st.sidebar:
    st.image("https://via.placeholder.com/300x100?text=MEGALITIKUM", use_container_width=True)
    st.markdown("### 📊 Performa Model")
    if interpreter is not None:
        st.metric("Test Accuracy", f"{model_info['test_accuracy']*100:.2f}%")
        st.metric("Best Validation", f"{model_info['best_val_accuracy_phase2']*100:.2f}%")
    else:
        st.error("Model tidak tersedia")
    st.markdown("---")
    st.markdown("### 🗿 Tentang Kelas")
    for nama in DESKRIPSI_KELAS.keys():
        with st.expander(nama):
            st.write(DESKRIPSI_KELAS[nama])

# ==============================================
# CEK MODEL
# ==============================================
if interpreter is None:
    st.error("""
    ❌ **Model tidak dapat dimuat!**
    
    Kemungkinan penyebab:
    1. FILE_ID belum diganti dengan ID Google Drive Anda
    2. File model belum diupload ke Google Drive
    3. Koneksi internet bermasalah
    
    **Cara memperbaiki:**
    1. Upload file `megalitikum_model.tflite` ke Google Drive
    2. Set sharing ke "Anyone with the link"
    3. Copy FILE_ID dari link
    4. Ganti variable FILE_ID di bagian atas kode
    """)
    st.stop()

# ==============================================
# TAB UTAMA
# ==============================================
tab1, tab2, tab3, tab4 = st.tabs(["📸 Prediksi", "ℹ️ Info Model", "📖 Panduan", "🔍 Filter Konten"])

# ==============================================
# TAB 1: PREDIKSI
# ==============================================
with tab1:
    st.markdown("### 📤 Ambil Gambar")
    
    # Layout responsif
    sumber = st.radio(
        "Pilih sumber gambar:",
        ["📁 Upload dari File", "📷 Ambil dengan Kamera"],
        horizontal=True
    )
    
    gambar = None
    if sumber == "📁 Upload dari File":
        gambar = st.file_uploader("Pilih file gambar...", type=['jpg', 'jpeg', 'png'])
    else:
        st.info("⚠️ Browser akan meminta izin akses kamera. Klik 'Allow' untuk melanjutkan.")
        gambar = st.camera_input("Ambil foto")

    if gambar:
        # Tampilkan gambar asli
        image = Image.open(gambar)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Gambar Asli", use_container_width=True)
        
        # ==========================================
        # CEK KUALITAS DAN DETEKSI OBJEK
        # ==========================================
        try:
            kualitas, pesan_kualitas, rekomendasi, blur_score, brightness, contrast = cek_kualitas_gambar(image)
        except:
            # Fallback jika scipy tidak tersedia
            kualitas = "Baik"
            pesan_kualitas = "Kualitas gambar baik"
            rekomendasi = ""
            blur_score = 0
            brightness = np.mean(np.array(image.convert('L')))
            contrast = np.std(np.array(image.convert('L')))
        
        # Deteksi objek non-batu
        is_non_megalith, deteksi_msg = detect_non_megalith(image)
        
        # Tampilkan peringatan kualitas
        if kualitas != "Baik":
            with col2:
                if kualitas == "Buruk":
                    st.error(f"⚠️ **Kualitas Gambar: {kualitas}**")
                else:
                    st.warning(f"⚠️ **Kualitas Gambar: {kualitas}**")
                
                if pesan_kualitas:
                    st.info(f"**Masalah:**\n{pesan_kualitas}")
                
                if rekomendasi:
                    st.caption(f"💡 **Enhancement:** {rekomendasi}")
        
        # Tampilkan peringatan objek non-batu
        if is_non_megalith:
            st.error(f"❌ **Deteksi Objek:** {deteksi_msg}")
            st.warning("""
            ⚠️ **Gambar ini terdeteksi BUKAN batu megalitikum!**
            
            Model hanya dilatih untuk mengklasifikasi BATU MEGALITIKUM.
            Prediksi untuk gambar non-batu akan TIDAK AKURAT.
            """)
            
            # Tanya apakah tetap lanjut
            lanjut = st.checkbox("Tetap lanjutkan prediksi? (Tidak disarankan)")
            if not lanjut:
                st.stop()
        
        # Tombol prediksi
        if st.button("🚀 Prediksi Sekarang", type="primary", use_container_width=True):
            
            with st.spinner("Menganalisis..."):
                if interpreter is None:
                    st.error("Model tidak tersedia.")
                    st.stop()
                
                # ======================================
                # ENHANCEMENT GAMBAR
                # ======================================
                enhanced_image = adaptive_enhancement(image, blur_score, brightness, contrast)
                
                # Tampilkan gambar enhanced
                with col2:
                    st.image(enhanced_image, caption="Gambar setelah Enhancement", use_container_width=True)
                
                # ======================================
                # PREDIKSI
                # ======================================
                predictions = predict_tflite(interpreter, input_details, output_details, enhanced_image)
                pred_idx = int(np.argmax(predictions))
                pred_class = class_names[pred_idx]
                confidence = float(predictions[pred_idx])

                top_3_idx = np.argsort(predictions)[-3:][::-1]
                top_3 = [(class_names[i], float(predictions[i])) for i in top_3_idx]

                # Tampilkan hasil
                st.success("### Hasil Prediksi")
                conf_color = "green" if confidence > 0.8 else "orange" if confidence > 0.6 else "red"
                
                # Tambahkan warning khusus
                warning_msg = ""
                if is_non_megalith:
                    warning_msg = "Gambar terdeteksi non-batu, prediksi TIDAK AKURAT!"
                    st.error(f"⚠️ {warning_msg}")
                elif confidence < 0.6:
                    warning_msg = "Confidence rendah, prediksi mungkin kurang akurat"
                    st.warning(f"⚠️ {warning_msg}")
                
                st.markdown(f"""
                <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-left: 4px solid #667eea;">
                    <h4>Kelas: <span style="color:#667eea;">{pred_class}</span></h4>
                    <p>Confidence: <span style="color:{conf_color};">{confidence:.2%}</span></p>
                </div>
                """, unsafe_allow_html=True)

                # Deskripsi
                st.info(f"**{pred_class}**: {DESKRIPSI_KELAS.get(pred_class, 'Tidak ada deskripsi.')}")

                # Top 3
                st.markdown("#### 🏆 Top 3")
                for i, (cls, conf) in enumerate(top_3, 1):
                    emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                    st.markdown(f"{emoji} **{i}. {cls}**: {conf:.2%}")

                # Grafik
                st.markdown("#### 📊 Probabilitas")
                sorted_idx = np.argsort(predictions)[::-1]
                chart_data = {
                    "Kelas": [class_names[i] for i in sorted_idx],
                    "Probabilitas": [float(predictions[i]) for i in sorted_idx]
                }
                st.bar_chart(chart_data, x="Kelas", y="Probabilitas", height=300)

                # Tombol download PDF
                pdf_bytes = buat_pdf_hasil(
                    gambar.name if hasattr(gambar, 'name') else "foto_kamera.jpg",
                    pred_class,
                    confidence,
                    top_3,
                    DESKRIPSI_KELAS.get(pred_class, ""),
                    kualitas,
                    warning_msg
                )
                st.download_button(
                    label="📥 Download Hasil (PDF)",
                    data=pdf_bytes,
                    file_name=f"hasil_{pred_class}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

# ==============================================
# TAB 2: INFO MODEL
# ==============================================
with tab2:
    st.markdown("### ℹ️ Detail Model")
    st.json({
        "arsitektur": "ResNet50 + Transfer Learning",
        "input_size": "224x224",
        "jumlah_kelas": 7,
        "test_accuracy": f"{model_info['test_accuracy']:.2%}",
        "best_validation": f"{model_info['best_val_accuracy_phase2']:.2%}",
        "format": "TFLite (float32)",
        "ukuran_file": "96.5 MB"
    })
    
    st.markdown("---")
    st.markdown("### ⚠️ Keterbatasan Model")
    st.warning("""
    **Model ini memiliki keterbatasan pada:**
    - Gambar dengan kualitas rendah (blur, gelap, terlalu terang)
    - Gambar dengan resolusi sangat kecil
    - Objek yang tidak terlihat jelas
    - Pencahayaan ekstrem
    - **Gambar NON-BATU** (hewan, tumbuhan, manusia, dll)
    
    **Saran penggunaan:**
    - Gunakan gambar dengan pencahayaan cukup
    - Pastikan objek terlihat jelas
    - Hindari gambar buram atau goyang
    - **HANYA gunakan gambar batu megalitikum**
    """)
    
    st.markdown("---")
    st.markdown("#### 🗿 Daftar Kelas")
    for nama, desk in DESKRIPSI_KELAS.items():
        st.markdown(f"**{nama}** : {desk}")

# ==============================================
# TAB 3: PANDUAN
# ==============================================
with tab3:
    st.markdown("### 📖 Cara Penggunaan")
    st.markdown("""
    1. **Pilih sumber gambar** (Upload file atau Kamera)
    2. **Ambil/upload gambar** batu megalitikum
    3. Klik tombol **Prediksi Sekarang**
    4. Lihat hasil klasifikasi, confidence, dan deskripsi
    5. **Download laporan PDF** jika diperlukan
    
    ### 💡 Tips Mendapatkan Hasil Akurat
    - Gunakan gambar dengan **pencahayaan cukup**
    - Pastikan **objek tidak blur**
    - Objek harus **terlihat jelas** di tengah frame
    - Hindari **background yang terlalu ramai**
    - Untuk hasil terbaik, gunakan gambar **resolusi tinggi**
    - **HANYA gunakan gambar batu megalitikum**
    
    ### ⚠️ Jika Hasil Tidak Akurat
    Aplikasi akan mendeteksi kualitas gambar dan memberi peringatan jika:
    - Gambar terlalu blur (akan di-sharpen otomatis)
    - Pencahayaan kurang/lebih (akan dinormalisasi)
    - Kontras rendah (akan ditingkatkan)
    - **Gambar terdeteksi sebagai NON-BATU** (akan ditolak)
    
    Confidence score di bawah 60% menandakan prediksi kurang yakin.
    """)
    
    st.info("Aplikasi ini dioptimalkan untuk tampilan mobile. Anda dapat mengakses semua fitur dengan mudah di ponsel.")

# ==============================================
# TAB 4: FILTER KONTEN
# ==============================================
with tab4:
    st.markdown("### 🔍 Filter Konten")
    st.markdown("""
    Aplikasi ini dilengkapi dengan **filter konten** untuk mendeteksi gambar NON-batu megalitikum.
    
    **Cara kerja filter:**
    - Analisis dominasi warna (hijau = tumbuhan, biru = langit/air)
    - Deteksi variasi warna (batu cenderung abu-abu seragam)
    - Analisis tekstur (batu memiliki tekstur khas)
    
    **Jika gambar terdeteksi sebagai NON-batu:**
    - Aplikasi akan menampilkan peringatan
    - Prediksi TIDAK akan akurat
    - Disarankan untuk tidak melanjutkan
    
    **Filter ini membantu:**
    - Mencegah prediksi salah pada gambar hewan/tumbuhan
    - Memberi edukasi kepada pengguna
    - Meningkatkan kepercayaan hasil
    """)
    
    # Demo deteksi
    st.markdown("### 📝 Demo Deteksi")
    st.write("Upload gambar untuk melihat cara kerja filter:")
    
    test_img = st.file_uploader("Upload gambar test", type=['jpg', 'jpeg', 'png'], key="test_filter")
    if test_img:
        test_image = Image.open(test_img)
        st.image(test_image, caption="Gambar Test", width=300)
        
        is_non, msg = detect_non_megalith(test_image)
        if is_non:
            st.error(f"❌ **HASIL:** {msg}")
        else:
            st.success(f"✅ **HASIL:** {msg}")

# ==============================================
# FOOTER
# ==============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Aplikasi Klasifikasi Batu Megalitikum<br>
    © 2024 - Seminar Hasil Penelitian</p>
</div>
""", unsafe_allow_html=True)
