import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from math import radians, cos, sin, asin, sqrt
import faiss
import io

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(
    page_title="Deteksi Usaha Mirip (SBR) – Faiss",
    layout="wide"
)

st.title("🔍 Deteksi & Pengelompokan Usaha Mirip (SBR) – Faiss")

# ===============================
# SIDEBAR – PARAMETER DINAMIS
# ===============================
st.sidebar.header("⚙️ Parameter Kemiripan")

TH_NAMA = st.sidebar.slider("Threshold Nama", 60, 100, 85)
TH_ALAMAT = st.sidebar.slider("Threshold Alamat", 60, 100, 75)
TH_JARAK_KUAT = st.sidebar.slider("Jarak Maksimum (meter)", 5, 50, 20)
SIM_THRESHOLD = st.sidebar.slider("Cosine Similarity (Faiss)", 0.50, 0.95, 0.75)
TOP_K = st.sidebar.slider("Jumlah Kandidat (TOP-K)", 3, 15, 5)
TOP_N_GROUP = st.sidebar.slider("Jumlah Kelompok Ditampilkan", 5, 50, 20)

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
    try:
        R = 6371000
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        return R * c
    except:
        return None

def skor_jarak(jarak):
    if jarak is None:
        return 50
    if jarak <= 10:
        return 100
    elif jarak <= 20:
        return 90
    elif jarak <= 50:
        return 70
    else:
        return 40

def skor_akhir(skor_nama, skor_alamat, jarak):
    return round(
        0.45 * skor_nama +
        0.35 * skor_alamat +
        0.20 * skor_jarak(jarak),
        2
    )

# ===============================
# UPLOAD FILE
# ===============================
uploaded_file = st.file_uploader("📤 Upload file Excel SBR", type=["xlsx"])
if not uploaded_file:
    st.info("Silakan upload file Excel terlebih dahulu.")
    st.stop()

df = pd.read_excel(uploaded_file)

# ===============================
# VALIDASI & KONVERSI
# ===============================
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

# ===============================
# PREPROCESSING
# ===============================
st.subheader("⚙️ Preprocessing Data")
df["nama_norm"] = df["nama_usaha"].apply(normalize_text)
df["alamat_norm"] = df["alamat_usaha"].apply(normalize_text)

# ===============================
# TF-IDF VECTOR (PROGRESS SENDIRI)
# ===============================
st.subheader("🧠 Membentuk vektor TF-IDF (Nama Usaha)")
progress_tfidf = st.progress(0)

vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=2)
X = vectorizer.fit_transform(df["nama_norm"].astype("U"))
progress_tfidf.progress(60)

X = X.astype("float32").toarray()
progress_tfidf.progress(100)

# ===============================
# FAISS INDEX (PROGRESS SENDIRI)
# ===============================
st.subheader("⚡ Mencari Kandidat Mirip (Faiss)")
progress_faiss = st.progress(0)

faiss.normalize_L2(X)
progress_faiss.progress(30)

index = faiss.IndexFlatIP(X.shape[1])
index.add(X)
progress_faiss.progress(70)

similarities, indices = index.search(X, TOP_K)
progress_faiss.progress(100)

# ===============================
# UNION FIND
# ===============================
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

# ===============================
# VALIDASI KANDIDAT (PROGRESS UTAMA)
# ===============================
st.subheader("🔗 Validasi Kandidat (Nama + Alamat + Jarak)")
progress_validate = st.progress(0)
total = len(df)

for i in range(len(df)):
    progress_validate.progress(int((i + 1) / total * 100))

    for pos, j in enumerate(indices[i]):
        if j <= i:
            continue
        if similarities[i][pos] < SIM_THRESHOLD:
            continue

        skor_n = fuzz.token_sort_ratio(df.loc[i, "nama_norm"], df.loc[j, "nama_norm"])
        skor_a = fuzz.token_sort_ratio(df.loc[i, "alamat_norm"], df.loc[j, "alamat_norm"])
        jarak = haversine(
            df.loc[i, "latitude"], df.loc[i, "longitude"],
            df.loc[j, "latitude"], df.loc[j, "longitude"]
        )

        final_score = skor_akhir(skor_n, skor_a, jarak)

        if (
            (skor_n >= TH_NAMA and skor_a >= TH_ALAMAT) or
            (skor_n >= TH_NAMA and jarak is not None and jarak <= TH_JARAK_KUAT)
        ):
            union(i, j)
            pair_scores.append({
                "idsbr_1": df.loc[i, "idsbr"],
                "idsbr_2": df.loc[j, "idsbr"],
                "skor_nama": skor_n,
                "skor_alamat": skor_a,
                "jarak_meter": round(jarak, 2) if jarak else None,
                "skor_akhir": final_score
            })

# ===============================
# PAIR DF + NAMA & ALAMAT
# ===============================
pair_df = pd.DataFrame(pair_scores)

if not pair_df.empty:
    pair_df = pair_df.merge(
        df[["idsbr", "nama_usaha", "alamat_usaha"]],
        left_on="idsbr_1", right_on="idsbr", how="left"
    ).rename(columns={
        "nama_usaha": "nama_usaha_1",
        "alamat_usaha": "alamat_usaha_1"
    }).drop(columns=["idsbr"])

    pair_df = pair_df.merge(
        df[["idsbr", "nama_usaha", "alamat_usaha"]],
        left_on="idsbr_2", right_on="idsbr", how="left"
    ).rename(columns={
        "nama_usaha": "nama_usaha_2",
        "alamat_usaha": "alamat_usaha_2"
    }).drop(columns=["idsbr"])

# ===============================
# BENTUK GRUP + CONFIDENCE
# ===============================
groups = defaultdict(list)
for idx in range(len(df)):
    groups[find(idx)].append(idx)

group_rows = []
for gid, members in groups.items():
    if len(members) > 1:
        skor_group = pair_df[
            pair_df["idsbr_1"].isin(df.loc[members, "idsbr"]) |
            pair_df["idsbr_2"].isin(df.loc[members, "idsbr"])
        ]["skor_akhir"]

        confidence = round(skor_group.mean(), 2) if not skor_group.empty else None

        group_rows.append({
            "group_id": f"G{gid}",
            "jumlah_usaha": len(members),
            "nama_representatif": df.loc[members[0], "nama_usaha"],
            "kecamatan": df.loc[members[0], "nmkec"],
            "confidence_group": confidence
        })

df_group = (
    pd.DataFrame(group_rows)
    .sort_values("jumlah_usaha", ascending=False)
    .reset_index(drop=True)
)

# ===============================
# RINGKASAN OTOMATIS
# ===============================
st.subheader("📊 Ringkasan Otomatis")
col1, col2, col3 = st.columns(3)

col1.metric("Total Usaha", len(df))
col2.metric("Total Kelompok Mirip", len(df_group))
col3.metric("Rata-rata Confidence", round(df_group["confidence_group"].mean(), 2))

# ===============================
# SEARCH + TOP N
# ===============================
st.subheader("📌 Kelompok Usaha Mirip")
search = st.text_input("🔎 Cari (nama usaha / kecamatan)")

df_view = df_group.copy()
if search:
    s = search.lower()
    df_view = df_view[
        df_view["nama_representatif"].str.lower().str.contains(s, na=False) |
        df_view["kecamatan"].str.lower().str.contains(s, na=False)
    ]

df_top = df_view.head(TOP_N_GROUP)
df_rest = df_view.iloc[TOP_N_GROUP:]

st.dataframe(df_top, use_container_width=True)

# ===============================
# EXPORT
# ===============================
if not df_rest.empty:
    buf_grp = io.BytesIO()
    df_rest.to_excel(buf_grp, index=False, engine="openpyxl")
    st.download_button(
        "📥 Download kelompok lainnya",
        data=buf_grp.getvalue(),
        file_name="kelompok_usaha_mirip_lainnya.xlsx"
    )

if not pair_df.empty:
    buf_pair = io.BytesIO()
    pair_df.to_excel(buf_pair, index=False, engine="openpyxl")
    st.download_button(
        "📥 Download pasangan usaha mirip",
        data=buf_pair.getvalue(),
        file_name="hasil_usaha_mirip.xlsx"
    )

# ===============================
# DETAIL PER GRUP (TOP N)
# ===============================
st.subheader("🔎 Detail Usaha per Kelompok")

for _, row in df_top.iterrows():
    gid = int(row["group_id"][1:])
    members = groups[gid]

    with st.expander(
        f"{row['group_id']} — {row['nama_representatif']} "
        f"({row['jumlah_usaha']} usaha | Confidence {row['confidence_group']})"
    ):
        detail = df.loc[members, [
            "idsbr", "nama_usaha", "alamat_usaha",
            "latitude", "longitude",
            "nmdesa", "nmkec",
            "status_perusahaan", "sumber_data"
        ]]
        st.dataframe(detail, use_container_width=True)

        st.markdown("#### 🧮 Skor Kemiripan Antar Usaha")
        st.dataframe(
            pair_df[
                (pair_df["idsbr_1"].isin(detail["idsbr"])) |
                (pair_df["idsbr_2"].isin(detail["idsbr"]))
            ][[
                "nama_usaha_1", "alamat_usaha_1",
                "nama_usaha_2", "alamat_usaha_2",
                "skor_nama", "skor_alamat",
                "jarak_meter", "skor_akhir"
            ]].sort_values("skor_akhir", ascending=False),
            use_container_width=True
        )

        st.markdown("#### 📍 Peta Lokasi")
        st.map(
            detail[["latitude", "longitude"]]
            .dropna()
            .rename(columns={"latitude": "lat", "longitude": "lon"})
        )

# ===============================
# LEGEND
# ===============================
st.markdown("""
### 🧭 Interpretasi Skor
- **Confidence ≥ 90** : Hampir pasti usaha sama  
- **80 – 89** : Sangat mungkin sama  
- **70 – 79** : Perlu verifikasi  
- **< 70** : Kemungkinan beda  
""")
