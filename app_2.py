# ==========================================================
# SBR SIMILARITY CHECK
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import time
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from math import radians, cos, sin, asin, sqrt
import io

# ===============================
# SESSION STATE
# ===============================
if "is_running" not in st.session_state:
    st.session_state.is_running = False

if "finished" not in st.session_state:
    st.session_state.finished = False

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

TH_NAMA = st.sidebar.slider(
    "Threshold Nama", 60, 100, 85,
    disabled=st.session_state.is_running, help=(
        "Kemiripan teks nama usaha (0–100).\n\n"
        "- Nilai tinggi maka pencocokan lebih ketat\n"
        "- Mengurangi salah deteksi\n"
        "- Bisa melewatkan duplikasi dengan penulisan berbeda\n\n"
        "Contoh:\n"
        "85 => 'Toko Sumber Rejeki' ≈ 'Sumber Rejeki'\n"
        "95 => Hampir identik\n\n"
        "Rekomendasi: 80–90"
    )
)
TH_ALAMAT = st.sidebar.slider(
    "Threshold Alamat", 60, 100, 75,
    disabled=st.session_state.is_running, help=(
        "Kemiripan teks alamat usaha.\n\n"
        "- Membantu membedakan usaha bernama sama\n"
        "- Alamat tidak baku bisa menurunkan skor\n\n"
        "Rekomendasi: 70–80"
    )
)
TH_JARAK_KUAT = st.sidebar.slider(
    "Jarak Maksimum (meter)", 5, 50, 20,
    disabled=st.session_state.is_running, help=(
        "Batas jarak koordinat GPS untuk dianggap usaha sama.\n\n"
        "- ≤ 10 m  : hampir pasti lokasi sama\n"
        "- ≤ 20 m  : sangat mungkin sama\n"
        "- ≤ 50 m  : masih mungkin (ruko/pasar)\n\n"
        "Rekomendasi: 20 meter"
    )
)
SIM_THRESHOLD = st.sidebar.slider(
    "Cosine Similarity", 0.50, 0.95, 0.75,
    disabled=st.session_state.is_running, help=(
        "Digunakan untuk menyaring kandidat awal saja.\n"
        "Bukan keputusan akhir.\n\n"
        "- Nilai tinggi maka kandidat lebih mirip\n"
        "- Proses lebih cepat\n"
        "- Bisa melewatkan pasangan yang agak berbeda\n\n"
        "Rekomendasi: 0.70 – 0.80"
    )
)
TOP_K = st.sidebar.slider(
    "Jumlah Kandidat (TOP-K)", 3, 15, 5,
    disabled=st.session_state.is_running, help=(
        "Jumlah tetangga terdekat yang dicek per usaha.\n\n"
        "- Nilai kecil maka lebih cepat tapi bisa miss\n"
        "- Nilai besar maka lebih lengkap tapi lebih lambat\n\n"
        "Rekomendasi: 5–10"
    )
)
TOP_N_GROUP = st.sidebar.slider(
    "Jumlah Kelompok Ditampilkan", 5, 50, 20,
    disabled=st.session_state.is_running, help=(
        "Jumlah kelompok usaha mirip yang ditampilkan di layar.\n"
        "Kelompok lain tetap bisa diunduh sebagai Excel."
    )
)

st.sidebar.markdown("---")

run_process = st.sidebar.button(
    "🚀 Terapkan & Proses",
    disabled=st.session_state.is_running
)

# ===============================
# LOGGING
# ===============================
st.subheader("🧾 Log Proses")
with st.expander("📜 Lihat log proses"):
    log_box = st.empty()
    logs = []

def log(msg):
    logs.append(msg)
    log_box.markdown("\n".join([f"- {l}" for l in logs[-10:]]))

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
log("File berhasil di-upload")

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

log(f"Total baris data: {len(df)}")

if not run_process:
    st.info("Atur parameter lalu klik **🚀 Terapkan & Proses**")
    st.stop()

st.session_state.is_running = True

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
log("Normalisasi teks selesai")

# ===============================
# TF-IDF (SPARSE)
# ===============================
st.subheader("🧠 TF-IDF Nama Usaha")
progress_tfidf = st.progress(0)

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 3),
    min_df=2
)
X = vectorizer.fit_transform(df["nama_norm"].astype("U"))

progress_tfidf.progress(100)
log("TF-IDF selesai (sparse)")

# ===============================
# KANDIDAT MIRIP (BATCH COSINE)
# ===============================
st.subheader("⚡ Kandidat Mirip (Batch Cosine)")
progress_cosine = st.progress(0)

N = X.shape[0]
similarities = np.zeros((N, TOP_K), dtype=np.float32)
indices = np.zeros((N, TOP_K), dtype=np.int32)

BATCH_SIZE = 200
for start in range(0, N, BATCH_SIZE):
    end = min(start + BATCH_SIZE, N)
    sim_chunk = cosine_similarity(X[start:end], X)

    for i in range(sim_chunk.shape[0]):
        row = sim_chunk[i]
        top_idx = np.argpartition(row, -TOP_K)[-TOP_K:]
        top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
        similarities[start + i] = row[top_idx]
        indices[start + i] = top_idx

    progress_cosine.progress(int(end / N * 100))

log("Pencarian kandidat selesai")

# ===============================
# VALIDASI + ETA
# ===============================
st.subheader("🔗 Validasi Kandidat")
progress_validate = st.progress(0)

parent = list(range(N))

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
start_time = time.time()

for i in range(N):
    progress = (i + 1) / N
    elapsed = time.time() - start_time
    eta = elapsed * (1 - progress) / progress if progress > 0 else 0

    progress_validate.progress(
        int(progress * 100),
        text=f"⏳ Sisa waktu: {int(eta//60)}m {int(eta%60)}s"
    )

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

            id1 = df.loc[i, "idsbr"]
            id2 = df.loc[j, "idsbr"]
            
            if id1 > id2:
                id1, id2 = id2, id1
                pair_scores.append({
                    "idsbr_1": df.loc[i, "idsbr"],
                    "idsbr_2": df.loc[j, "idsbr"],
                    "skor_nama": sn,
                    "skor_alamat": sa,
                    "jarak_meter": round(jarak, 2) if jarak else None,
                    "skor_akhir": skor_akhir(sn, sa, jarak)
                })

log(f"Total pasangan mirip: {len(pair_scores)}")

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
# GROUP + CONFIDENCE
# ===============================
groups = defaultdict(list)
for i in range(N):
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

st.session_state.df_group = df_group
st.session_state.pair_df = pair_df
st.session_state.groups = groups
st.session_state.df = df

# ===============================
# RINGKASAN
# ===============================
st.subheader("📊 Ringkasan Otomatis")
c1, c2, c3 = st.columns(3)
c1.metric("Total Usaha", len(df))
c2.metric("Total Kelompok Mirip", len(df_group))
c3.metric("Rata-rata Confidence", round(df_group["confidence_group"].mean(), 2))

# ===============================
# TAMPILAN HASIL
# ===============================
st.subheader("📌 Kelompok Usaha Mirip")
search = st.text_input("🔎 Cari (nama usaha / kecamatan)")

df_view = st.session_state.df_group.copy()
if search:
    s = search.lower()
    df_view = df_view[
        df_view["nama_representatif"].str.lower().str.contains(s, na=False) |
        df_view["kecamatan"].str.lower().str.contains(s, na=False)
    ]

df_top = df_view.head(TOP_N_GROUP)
df_rest = df_view.iloc[TOP_N_GROUP:]

st.dataframe(df_top, use_container_width=True)

st.session_state.is_running = False
st.session_state.finished = True
st.success("✅ Proses selesai")

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
        detail = st.session_state.df.loc[members, [
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
                "idsbr_1","nama_usaha_1", "alamat_usaha_1",
                "idsbr_2","nama_usaha_2", "alamat_usaha_2",
                "skor_nama", "skor_alamat",
                "jarak_meter", "skor_akhir"
            ]].sort_values("skor_akhir", ascending=False),
            use_container_width=True
        )
        st.caption(
            f"Kelompok ini berisi {len(set(pair_df['idsbr_1']).union(pair_df['idsbr_2']))} usaha unik "
            f"dengan {len(pair_df)} pasangan kemiripan"
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

