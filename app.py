import streamlit as st
import pickle
import pandas as pd
import numpy as np

# =========================
# CONFIG & BASIC STYLING
# =========================
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# Sedikit CSS biar lebih enak dilihat
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fb;
    }
    .stSidebar {
        background-color: #101827;
    }
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f9fafb;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #374151;
        margin-bottom: 0.75rem;
    }
    .sidebar-section-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #d1d5db;
        margin-top: 1rem;
        margin-bottom: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .card {
        padding: 1.5rem;
        border-radius: 14px;
        background-color: #ffffff;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
        border: 1px solid #e5e7eb;
    }
    .card-header {
        font-size: 0.95rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.75rem;
    }
    .price-text {
        font-size: 2rem;
        font-weight: 700;
        color: #16a34a;
        margin-bottom: 0.35rem;
    }
    .badge {
        display: inline-block;
        font-size: 0.75rem;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        background-color: #e0f2fe;
        color: #0369a1;
        margin-right: 0.25rem;
        margin-bottom: 0.25rem;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .section-subtitle {
        font-size: 0.85rem;
        color: #6b7280;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Pemuatan Model ---
MODEL_FILE = 'LGBMRegressor_best_model.pkl'
try:
    with open(MODEL_FILE, 'rb') as f:
        prediction_model = pickle.load(f)
except FileNotFoundError:
    st.error("❌ File model 'LGBMRegressor_best_model.pkl' tidak ditemukan.")
    st.stop()

# ======================================================================
# KONFIGURASI (SAMA SEPERTI KODE AWALMU)
# ======================================================================

# 1. OPSI KATEGORIKAL
make_options = ['Kia', 'BMW', 'Volvo', 'Audi', 'Nissan', 'Hyundai', 'Chevrolet', 'Ford', 'Acura', 'Cadillac', 'Infiniti', 'Lincoln', 'Jeep', 'Mercedes-Benz', 'GMC', 'Dodge', 'Honda', 'Chrysler', 'Ram', 'Lexus', 'Subaru', 'Mazda', 'Toyota', 'Volkswagen', 'Buick', 'Maserati', 'Land Rover', 'Porsche', 'Jaguar', 'Mitsubishi']
model_options = ['Sorento', '3 Series', 'S60', 'A3', 'Altima', 'Elantra', 'Cruze', 'F-150', 'MDX', 'CTS', 'G37', 'MKZ', 'Grand Cherokee', 'E-Class', 'Acadia', 'Charger', 'Civic', 'Town and Country', '1500', 'IS 250', 'Outback', 'Mazda3', 'Corolla', 'Jetta', 'Enclave', 'Ghibli', 'Range Rover', 'Cayenne', 'XF', 'Outlander Sport']
trim_options = ['LX', 'Base', 'T5', 'Premium', '2.5 S', 'SE', '1LT', 'XLT', 'i', 'Luxury', 'Journey', 'Hybrid', 'Laredo', 'E350', 'SLE', 'SXT', 'EX', 'Touring', 'Big Horn', 'Sport', '2.5i Premium', 's Grand Touring', 'L', 'SportWagen SE', 'Convenience', 'Limited', 'LTZ', 'SLT', 'Express', 'SR5', 'ES 350']
body_options = ['SUV', 'Sedan', 'Wagon', 'Convertible', 'Coupe', 'Hatchback', 'Crew Cab', 'Minivan', 'Van', 'SuperCrew', 'SuperCab', 'Quad Cab', 'King Cab', 'Double Cab', 'Extended Cab', 'Access Cab']
state_options = ['fl', 'ca', 'pa', 'tx', 'ga', 'in', 'nj', 'va', 'il', 'tn', 'az', 'oh', 'mi', 'nc', 'co', 'sc', 'mo', 'md', 'wi', 'nv', 'ma', 'pr', 'mn', 'or', 'wa', 'ny', 'la', 'hi', 'ne', 'ut', 'al', 'ms', 'ct']
transmission_options = ['automatic', 'manual', 'others']
color_options = ['black', 'white', 'gray', 'silver', 'blue', 'red', '—']  # LENGKAPI bila perlu
interior_options = ['black', 'gray', 'beige', 'tan', 'brown', '—']        # LENGKAPI bila perlu
seller_options = [
    'nissan infiniti of honolulu',
    'the hertz corporation',
    'ford motor credit company,llc'
]  # LENGKAPI bila perlu

# 2. PEMETAAN LABEL ENCODER
make_map = {label: i for i, label in enumerate(make_options)}
model_map = {label: i for i, label in enumerate(model_options)}
trim_map = {label: i for i, label in enumerate(trim_options)}
body_map = {label: i for i, label in enumerate(body_options)}
state_map = {label: i for i, label in enumerate(state_options)}
color_map = {label: i for i, label in enumerate(color_options)}
interior_map = {label: i for i, label in enumerate(interior_options)}
seller_map = {label: i for i, label in enumerate(seller_options)}

# 3. NILAI MEAN & SCALE DARI STANDARDSCALER (ISI DARI NOTEBOOK)
numerical_features = ['year', 'condition', 'odometer', 'mmr']
scaler_means = {
    'year': 2010.0,
    'condition': 3.0,
    'odometer': 60000.0,
    'mmr': 15000.0
}  # GANTI DENGAN NILAI ASLI
scaler_scales = {
    'year': 2.5,
    'condition': 1.2,
    'odometer': 35000.0,
    'mmr': 8000.0
}  # GANTI DENGAN NILAI ASLI

# 4. URUTAN KOLOM FINAL
TRAINING_COLUMN_ORDER = [
    'year', 'condition', 'odometer', 'mmr', 'make', 'model', 'trim', 'body',
    'state', 'color', 'interior', 'seller', 'transmission_automatic',
    'transmission_manual', 'transmission_others'
]

# =========================
# FUNGSI PREPROCESSING
# =========================
def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing manual agar sama dengan training di notebook."""
    df_processed = df.copy()

    # 1. Label Encoding
    df_processed['make'] = df_processed['make'].map(make_map)
    df_processed['model'] = df_processed['model'].map(model_map)
    df_processed['trim'] = df_processed['trim'].map(trim_map)
    df_processed['body'] = df_processed['body'].map(body_map)
    df_processed['state'] = df_processed['state'].map(state_map)
    df_processed['color'] = df_processed['color'].map(color_map)
    df_processed['interior'] = df_processed['interior'].map(interior_map)
    df_processed['seller'] = df_processed['seller'].map(seller_map)

    # 2. One-Hot untuk transmisi
    for option in transmission_options:
        col_name = f"transmission_{option}"
        df_processed[col_name] = (df_processed['transmission'] == option).astype(int)
    df_processed = df_processed.drop('transmission', axis=1)

    # 3. Standard Scaling numerik
    for col in numerical_features:
        df_processed[col] = (df_processed[col] - scaler_means[col]) / scaler_scales[col]

    # 4. Urutan kolom sesuai training
    df_processed = df_processed[TRAINING_COLUMN_ORDER]

    return df_processed


# =========================
# SIDEBAR – INPUT USER
# =========================
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙️ Konfigurasi Fitur Mobil</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">Spesifikasi Utama</div>', unsafe_allow_html=True)
    year = st.slider('Tahun', 1982, 2015, 2010)
    condition = st.slider('Kondisi (1-50)', 1.0, 50.0, 25.0)
    odometer = st.number_input('Odometer', value=68000.0, step=1000.0)
    mmr = st.number_input('MMR (Nilai Pasar)', value=13000.0, step=500.0)

    st.markdown('<div class="sidebar-section-title">Detail Mobil</div>', unsafe_allow_html=True)
    make = st.selectbox('Merek', make_options)
    model = st.selectbox('Model', model_options)
    trim = st.selectbox('Trim', trim_options)
    body = st.selectbox('Tipe Bodi', body_options)
    transmission = st.selectbox('Transmisi', transmission_options)

    st.markdown('<div class="sidebar-section-title">Lokasi & Warna</div>', unsafe_allow_html=True)
    state = st.selectbox('Negara Bagian', state_options)
    color = st.selectbox('Warna Eksterior', color_options)
    interior = st.selectbox('Interior', interior_options)

    st.markdown('<div class="sidebar-section-title">Penjual</div>', unsafe_allow_html=True)
    seller = st.selectbox('Penjual', seller_options)

    predict_clicked = st.button('🚀 Prediksi Harga Jual')


# =========================
# MAIN LAYOUT
# =========================
st.markdown(
    """
    <div class="section-title">Prediksi Harga Mobil</div>
    <div class="section-subtitle">
        Masukkan spesifikasi mobil di sidebar kiri, lalu klik tombol <b>“🚀 Prediksi Harga Jual”</b> untuk melihat estimasi harga.
    </div>
    """,
    unsafe_allow_html=True,
)

# Susun data input
input_data = {
    'year': year,
    'condition': condition,
    'odometer': odometer,
    'mmr': mmr,
    'make': make,
    'model': model,
    'trim': trim,
    'body': body,
    'transmission': transmission,
    'state': state,
    'color': color,
    'interior': interior,
    'seller': seller
}
input_df = pd.DataFrame(input_data, index=[0])

col_left, col_right = st.columns([1.6, 1.4])

with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Detail Input</div>', unsafe_allow_html=True)

    # Tampilkan info penting sebagai "badge"
    st.markdown(
        f"""
        <div style="margin-bottom: 0.5rem;">
            <span class="badge">{make} {model}</span>
            <span class="badge">{trim}</span>
            <span class="badge">{body}</span>
            <span class="badge">{year}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("### Spesifikasi")
    st.write(
        f"""
        - **Tahun:** {year}  
        - **Kondisi:** {condition}  
        - **Odometer:** {odometer:,.0f} km  
        - **MMR (Market Value):** ${mmr:,.0f}  
        """
    )

    st.write("### Detail Lainnya")
    st.write(
        f"""
        - **Transmisi:** {transmission}  
        - **Warna / Interior:** {color} / {interior}  
        - **Lokasi (State):** {state.upper()}  
        - **Penjual:** {seller}  
        """
    )

    with st.expander("Lihat Data Mentah yang Dikirim ke Model"):
        st.dataframe(input_df)

    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Prediksi Harga Jual</div>', unsafe_allow_html=True)

    if predict_clicked:
        try:
            input_df_processed = preprocess_input(input_df)
            prediction = prediction_model.predict(input_df_processed)

            st.markdown(f'<div class="price-text">${prediction[0]:,.2f}</div>', unsafe_allow_html=True)
            st.markdown(
                """
                <div style="font-size: 0.85rem; color: #6b7280; margin-bottom: 0.5rem;">
                    Estimasi harga jual berdasarkan model regresi yang telah dilatih.
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress( min(1.0, max(0.0, prediction[0] / 50000.0)) )

            st.caption("Catatan: Skala harga hanya ilustrasi, sesuaikan dengan karakteristik dataset aslinya.")
        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")
            st.warning("Pastikan semua daftar opsi dan nilai scaler di bagian KONFIGURASI sudah diisi dengan benar.")
    else:
        st.info("Masukkan fitur di sidebar lalu klik tombol **🚀 Prediksi Harga Jual** untuk melihat hasil.")

    st.markdown('</div>', unsafe_allow_html=True)
