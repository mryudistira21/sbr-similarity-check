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

st.title("🔍 Deteksi & Pengelompokan Usaha Mirip (SBR)")

# ===============================
# SIDEBAR – PARAMETER + GOVERNANCE
# ===============================
st.sidebar.header("⚙️ Parameter Kemiripan")

TH_NAMA = st.sidebar.slider(
    "Threshold Nama",
    60, 100, 85,
    help=(
        "Kemiripan teks nama usaha (0–100).\n\n"
        "• Nilai tinggi → pencocokan lebih ketat\n"
        "• Mengurangi salah deteksi\n"
        "• Bisa melewatkan duplikasi dengan penulisan berbeda\n\n"
        "Contoh:\n"
        "85 → 'Toko Sumber Rejeki' ≈ 'Sumber Rejeki'\n"
        "95 → Hampir identik\n\n"
        "Rekomendasi: 80–90"
    )
)

TH_ALAMAT = st.sidebar.slider(
    "Threshold Alamat",
    60, 100, 75,
    help=(
        "Kemiripan teks alamat usaha.\n\n"
        "• Membantu membedakan usaha bernama sama\n"
        "• Alamat tidak baku bisa menurunkan skor\n\n"
        "Rekomendasi: 70–80"
    )
)

TH_JARAK_KUAT = st.sidebar.slider(
    "Jarak Maksimum (meter)",
    5, 50, 20,
    help=(
        "Batas jarak koordinat GPS untuk dianggap usaha sama.\n\n"
        "• ≤ 10 m  : hampir pasti lokasi sama\n"
        "• ≤ 20 m  : sangat mungkin sama\n"
        "• ≤ 50 m  : masih mungkin (ruko/pasar)\n\n"
        "Rekomendasi: 20 meter"
    )
)

SIM_THRESHOLD = st.sidebar.slider(
    "Cosine Similarity (Faiss)",
    0.50, 0.95, 0.75,
    help=(
        "Digunakan untuk menyaring kandidat awal saja.\n"
        "Bukan keputusan akhir.\n\n"
        "• Nilai tinggi → kandidat lebih mirip\n"
        "• Proses lebih cepat\n"
        "• Bisa melewatkan pasangan yang agak berbeda\n\n"
        "Rekomendasi: 0.70 – 0.80"
    )
)

TOP_K = st.sidebar.slider(
    "Jumlah Kandidat (TOP-K)",
    3, 15, 5,
    help=(
        "Jumlah tetangga terdekat yang dicek per usaha.\n\n"
        "• Nilai kecil → cepat tapi bisa miss\n"
        "• Nilai besar → lebih lengkap tapi lebih lambat\n\n"
        "Rekomendasi: 5–10"
    )
)

TOP_N_GROUP = st.sidebar.slider(
    "Jumlah Kelompok Ditampilkan",
    5, 50, 20,
    help=(
        "Jumlah kelompok usaha mirip yang ditampilkan di layar.\n"
        "Kelompok lain tetap bisa diunduh sebagai Excel."
    )
)

st.sidebar.markdown("---")
run_process = st.sidebar.button("🚀 Terapkan & Proses")

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
st.subheader("🧠 TF-IDF Nama Usaha")
p1 = st.progress(0)
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=2)
X = vectorizer.fit_transform(df["nama_norm"].astype("U")).astype("float32").toarray()
p1.progress(100)

# ===============================
# FAISS
# ===============================
st.subheader("⚡ Kandidat Mirip (Faiss)")
p2 = st.progress(0)
faiss.normalize_L2(X)
index = faiss.IndexFlatIP(X.shape[1])
index.add(X)
similarities, indices = index.search(X, TOP_K)
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
# PAIR DF + DETAIL
# ===============================
pair_df = pd.DataFrame(pair_scores)

if not pair_df.empty:
    for side in ["1", "2"]:
        pair_df = pair_df.merge(
            df[["idsbr", "nama_usaha", "alamat_usaha"]],
            left_on=f"idsbr_{side}",
            right_on="idsbr",
            how="left"
        ).rename(columns={
            "nama_usaha": f"nama_usaha_{side}",
            "alamat_usaha": f"alamat_usaha_{side}"
        }).drop(columns=["idsbr"])

# ===============================
# GROUP + CONFIDENCE
# ===============================
groups = defaultdict(list)
for i in range(len(df)):
    groups[find(i)].append(i)

group_rows = []
for gid, members in groups.items():
    if len(members) > 1:
        skor_group = pair_df[
            pair_df["idsbr_1"].isin(df.loc[members, "idsbr"]) |
            pair_df["idsbr_2"].isin(df.loc[members, "idsbr"])
        ]["skor_akhir"]
        group_rows.append({
            "group_id": f"G{gid}",
            "jumlah_usaha": len(members),
            "nama_representatif": df.loc[members[0], "nama_usaha"],
            "kecamatan": df.loc[members[0], "nmkec"],
            "confidence_group": round(skor_group.mean(), 2) if not skor_group.empty else None
        })

df_group = pd.DataFrame(group_rows).sort_values("jumlah_usaha", ascending=False)

# ===============================
# RINGKASAN OTOMATIS
# ===============================
st.subheader("📊 Ringkasan Otomatis")
c1, c2, c3 = st.columns(3)
c1.metric("Total Usaha", len(df))
c2.metric("Total Kelompok Mirip", len(df_group))
c3.metric("Rata-rata Confidence", round(df_group["confidence_group"].mean(), 2))

# ===============================
# STATISTIK KECAMATAN
# ===============================
st.subheader("📍 Statistik Lintas Kecamatan")
kec = df_group.groupby("kecamatan").agg(
    jumlah_kelompok=("group_id", "count"),
    total_usaha=("jumlah_usaha", "sum"),
    rata_confidence=("confidence_group", "mean")
).reset_index().sort_values("jumlah_kelompok", ascending=False)

st.dataframe(kec, use_container_width=True)

st.markdown("""
**Interpretasi:**
- Kecamatan dengan **banyak kelompok + confidence tinggi** adalah **prioritas verifikasi lapangan**.
""")

# ===============================
# SEARCH + TOP N
# ===============================
st.subheader("📌 Kelompok Usaha Mirip")
search = st.text_input("🔎 Cari (nama / kecamatan)")

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
# DETAIL PER GRUP
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
- **≥ 90** : Hampir pasti usaha sama  
- **80 – 89** : Sangat mungkin sama  
- **70 – 79** : Perlu verifikasi  
- **< 70** : Kemungkinan beda  
""")

