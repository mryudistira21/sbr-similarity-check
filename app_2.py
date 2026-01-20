# ==========================================================
# FAISS OPTIONAL (AMAN UNTUK STREAMLIT CLOUD)
# ==========================================================
USE_FAISS = False
try:
    import faiss
    USE_FAISS = True
except Exception:
    USE_FAISS = False

import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from math import radians, cos, sin, asin, sqrt
import io

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(
    page_title="Deteksi Usaha Mirip (SBR)",
    layout="wide"
)

st.title("🔍 Deteksi & Pengelompokan Usaha Mirip (SBR)")

# ===============================
# SIDEBAR – PARAMETER
# ===============================
st.sidebar.header("⚙️ Parameter Kemiripan")

TH_NAMA = st.sidebar.slider("Threshold Nama", 60, 100, 85)
TH_ALAMAT = st.sidebar.slider("Threshold Alamat", 60, 100, 75)
TH_JARAK_KUAT = st.sidebar.slider("Jarak Maksimum (meter)", 5, 50, 20)
SIM_THRESHOLD = st.sidebar.slider("Cosine Similarity", 0.50, 0.95, 0.75)
TOP_K = st.sidebar.slider("Jumlah Kandidat (TOP-K)", 3, 15, 5)
TOP_N_GROUP = st.sidebar.slider("Jumlah Kelompok Ditampilkan", 5, 50, 20)

st.sidebar.markdown("---")
run_process = st.sidebar.button("🚀 Terapkan & Proses")

st.sidebar.info(
    f"Mode pencarian kandidat: **{'FAISS (cepat)' if USE_FAISS else 'Cosine Fallback (aman)'}**"
)

# ===============================
# FUNGSI UTIL
# ===============================
def normalize_text(text):
    if not isinstance(text, str):
        return ""
    return " ".join(text.lower().strip().split())

def to_float(val):
    try:
        if pd.isna(val):
            return None
        if isinstance(val, str):
            val = val.replace(",", ".").strip()
        return float(val)
    except:
        return None

def haversine(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2):
        return None
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def skor_jarak(jarak):
    if jarak is None:
        return 50
    if jarak <= 10:
        return 100
    elif jarak <= 20:
        return 90
    elif jarak <= 50:
        return 70
    return 50

def skor_akhir(sn, sa, jarak):
    return round(0.45*sn + 0.35*sa + 0.20*skor_jarak(jarak), 2)

# ===============================
# UPLOAD FILE
# ===============================
uploaded_file = st.file_uploader("📤 Upload file Excel SBR", type=["xlsx"])
if not uploaded_file:
    st.stop()

df = pd.read_excel(uploaded_file)

required_cols = [
    "idsbr", "nama_usaha", "alamat_usaha",
    "latitude", "longitude",
    "nmkec", "nmdesa",
    "status_perusahaan", "sumber_data"
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Kolom tidak ditemukan: {missing}")
    st.stop()

df["latitude"] = df["latitude"].apply(to_float)
df["longitude"] = df["longitude"].apply(to_float)

if not run_process:
    st.info("Atur parameter di sidebar lalu klik **🚀 Terapkan & Proses**")
    st.stop()

# ===============================
# PREPROCESSING
# ===============================
st.subheader("⚙️ Preprocessing Data")
df["nama_norm"] = df["nama_usaha"].apply(normalize_text)
df["alamat_norm"] = df["alamat_usaha"].apply(normalize_text)

# ===============================
# TF-IDF
# ===============================
st.write("Jumlah baris:", len(df))

st.subheader("🧠 TF-IDF Nama Usaha")
p1 = st.progress(0)
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=2)
X = vectorizer.fit_transform(df["nama_norm"].astype("U"))
p1.progress(100)

# ===============================
# KANDIDAT MIRIP (FAISS / FALLBACK)
# ===============================
st.subheader("⚡ Kandidat Mirip")

p2 = st.progress(0)

N = X.shape[0]

similarities = np.zeros((N, TOP_K), dtype=np.float32)
indices = np.zeros((N, TOP_K), dtype=np.int32)

BATCH_SIZE = 200  # kecil tapi aman

for start in range(0, X.shape[0], BATCH_SIZE):
    end = min(start + BATCH_SIZE, X.shape[0])

    sim_chunk = cosine_similarity(X[start:end], X)

    for i in range(sim_chunk.shape[0]):
        row = sim_chunk[i]
        top_idx = np.argpartition(row, -TOP_K)[-TOP_K:]
        top_idx = top_idx[np.argsort(row[top_idx])[::-1]]

        similarities[start + i] = row[top_idx]
        indices[start + i] = top_idx

p2.progress(100)

# ===============================
# VALIDASI KANDIDAT
# ===============================
st.subheader("🔗 Validasi Kandidat")
p3 = st.progress(0)

parent = list(range(len(df)))

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

pair_scores = []

for i in range(len(df)):
    p3.progress(int((i + 1) / len(df) * 100))
    for pos, j in enumerate(indices[i]):
        if j <= i or similarities[i][pos] < SIM_THRESHOLD:
            continue

        sn = fuzz.token_sort_ratio(df.loc[i, "nama_norm"], df.loc[j, "nama_norm"])
        sa = fuzz.token_sort_ratio(df.loc[i, "alamat_norm"], df.loc[j, "alamat_norm"])
        jarak = haversine(
            df.loc[i, "latitude"], df.loc[i, "longitude"],
            df.loc[j, "latitude"], df.loc[j, "longitude"]
        )

        if (sn >= TH_NAMA and sa >= TH_ALAMAT) or (sn >= TH_NAMA and jarak and jarak <= TH_JARAK_KUAT):
            union(i, j)
            pair_scores.append({
                "idsbr_1": df.loc[i, "idsbr"],
                "idsbr_2": df.loc[j, "idsbr"],
                "skor_nama": sn,
                "skor_alamat": sa,
                "jarak_meter": round(jarak, 2) if jarak else None,
                "skor_akhir": skor_akhir(sn, sa, jarak)
            })

# ===============================
# HASIL
# ===============================
pair_df = pd.DataFrame(pair_scores)

groups = defaultdict(list)
for i in range(len(df)):
    groups[find(i)].append(i)

group_rows = []
for gid, members in groups.items():
    if len(members) > 1:
        skor = pair_df[
            pair_df["idsbr_1"].isin(df.loc[members, "idsbr"]) |
            pair_df["idsbr_2"].isin(df.loc[members, "idsbr"])
        ]["skor_akhir"]
        group_rows.append({
            "group_id": f"G{gid}",
            "jumlah_usaha": len(members),
            "nama_representatif": df.loc[members[0], "nama_usaha"],
            "kecamatan": df.loc[members[0], "nmkec"],
            "confidence_group": round(skor.mean(), 2) if not skor.empty else None
        })

df_group = pd.DataFrame(group_rows).sort_values("jumlah_usaha", ascending=False)

st.success("✅ Proses selesai tanpa crash")
st.dataframe(df_group.head(TOP_N_GROUP), use_container_width=True)



