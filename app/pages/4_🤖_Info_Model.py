import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st

from app.data import days_since, feature_daily_row_count, load_model_metadata
from app.style import inject_base_css, render_developer_footer
from engine.decision import BUY_THRESHOLD
from engine.predict_turnaround import MODEL_VERSION as TURNAROUND_MODEL_VERSION

st.set_page_config(page_title="MyStocks — Info Model", page_icon="🤖", layout="wide")
inject_base_css()
render_developer_footer()

if st.button("← Kembali ke Home"):
    st.switch_page("Home.py")

st.title("🤖 Info Model")
st.caption("Detail kedua model machine learning yang dipakai aplikasi ini -- Swing dan Turnaround.")


def render_header(meta: dict, base_rate_label: str) -> None:
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.markdown("<span class='mystocks-muted'>Model</span>", unsafe_allow_html=True)
        st.markdown(f"<div class='mystocks-metric-value'>{meta['model_version']}</div>", unsafe_allow_html=True)
    c2.metric("Dilatih pada", meta["trained_at"], delta=f"{days_since(meta['trained_at'])} hari lalu", delta_color="off")
    c3.metric(base_rate_label, f"{meta['base_rate']*100:.1f}%")


def render_retrain_reminder(meta: dict, current_rows: int, *, comparable: bool = True) -> None:
    trained_rows = meta["n_training_rows"]
    st.subheader("📈 Pengingat retrain")
    if not comparable:
        st.caption(
            f"Dilatih dengan {trained_rows:,} baris kandidat (subset yang sedang bearish/bottoming saat "
            "training, bukan seluruh histori) -- tidak dibandingkan otomatis dengan jumlah baris "
            "feature_daily saat ini karena basisnya beda (subset vs total), bukan perbandingan apel-ke-apel."
        )
        return
    new_rows = current_rows - trained_rows
    d1, d2, d3 = st.columns(3)
    d1.metric("Baris data saat training", f"{trained_rows:,}")
    d2.metric("Baris data sekarang", f"{current_rows:,}")
    d3.metric("Data baru sejak training", f"{new_rows:,}", delta=f"+{new_rows:,}" if new_rows > 0 else None)
    if new_rows > trained_rows * 0.2:
        st.info(
            f"Data sudah bertambah {new_rows:,} baris ({new_rows/trained_rows*100:.0f}%) sejak model ini "
            "dilatih. Pertimbangkan retrain manual di notebook Colab (`notebooks/fase3_ml_research.ipynb`) "
            "dengan data terbaru, lalu commit model baru ke `models/`."
        )
    else:
        st.caption("Belum ada indikasi kuat untuk retrain berdasarkan volume data baru.")


RETRAIN_DISCLAIMER = (
    "Retrain sengaja tidak disediakan sebagai tombol di aplikasi ini -- proses training butuh review "
    "manusia atas walk-forward validation sebelum model baru dipercaya. Halaman ini hanya menampilkan "
    "indikator, bukan memicu training."
)


def render_features_and_tickers(meta: dict) -> None:
    st.subheader(f"🧬 Fitur yang dipakai model ({len(meta['feature_cols'])})")
    st.dataframe({"feature": meta["feature_cols"]}, width="stretch", hide_index=True, height=300)
    st.subheader("🏢 Ticker dalam data training")
    st.write(", ".join(meta.get("tickers") or []))


current_rows = feature_daily_row_count()

# ============================================================ SWING =====
st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)
st.header("🎯 Model Swing (10 hari)")

meta = load_model_metadata()
render_header(meta, "Baseline win rate (base_rate)")
st.caption(
    "Ini angka **pembanding acak** (kalau asal pilih saham tanpa strategi apa pun, kira-kira segini "
    "sering yang naik ≥5% sebelum -2.5% dalam 10 hari) -- bukan skor performa model. Model dianggap "
    "bekerja kalau bisa MENGGESER probabilitas jauh dari angka ini: BUY seharusnya jauh di atas, "
    "AVOID jauh di bawah. Lihat halaman **Swing** untuk win rate sesungguhnya per keputusan."
)

st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)
render_retrain_reminder(meta, current_rows)
st.caption(RETRAIN_DISCLAIMER)

st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)
st.subheader("🎯 Definisi target & aturan keputusan")
t1, t2, t3 = st.columns(3)
t1.metric("Target profit", f"{meta['target_pct']*100:.1f}%")
t2.metric("Stop loss", f"{meta['stop_pct']*100:.1f}%")
t3.metric("Horizon", f"{meta['horizon_days']} hari trading")
st.markdown(
    f"""
    - **BUY** — probabilitas ≥ {BUY_THRESHOLD*100:.0f}%
    - **WATCH** — probabilitas ≥ base rate historis model, tapi < {BUY_THRESHOLD*100:.0f}%
    - **AVOID** — probabilitas di bawah base rate (tidak ada edge), atau harga saham ≤ Rp50 (floor gocap,
      lihat `engine.decision.GOCAP_PRICE_FLOOR` — dipaksa AVOID apa pun probabilitasnya)
    """
)

st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)
render_features_and_tickers(meta)

# ========================================================= TURNAROUND ===
st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)
st.header("🔄 Model Turnaround")

try:
    ta_meta = load_model_metadata(TURNAROUND_MODEL_VERSION)
except FileNotFoundError:
    st.caption("Model turnaround belum dilatih.")
else:
    ta_wf = ta_meta.get("walk_forward_validation") or {}
    ta_ml = ta_wf.get("avg_ml_metrics") or {}
    ta_threshold = ta_wf.get("buy_threshold", 0)

    render_header(ta_meta, "Base rate (kandidat bearish/bottoming)")
    tb1, tb2, tb3 = st.columns(3)
    tb1.metric("Threshold POTENSIAL", f"{ta_threshold*100:.0f}%")
    tb2.metric("Precision @ threshold (walk-forward)", f"{ta_ml.get('precision', 0)*100:.1f}%" if ta_ml else "-")
    tb3.metric("ROC-AUC (walk-forward)", f"{ta_ml.get('roc_auc', 0):.3f}" if ta_ml else "-")
    st.caption(
        "Beda karakter dari model Swing di atas: base rate-nya sendiri sudah tinggi (~82%), jadi lift "
        "di atas base rate lebih kecil (precision ~92% vs base 82%, bukan lompatan besar seperti BUY "
        "swing vs base rate 30%-nya). Nilainya lebih ke menyisihkan kandidat yang kemungkinan besar "
        "GAGAL berbalik arah, bukan menemukan yang pasti berhasil. Lihat halaman **Turnaround** untuk "
        "kandidat yang sedang aktif."
    )

    st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)
    render_retrain_reminder(ta_meta, current_rows, comparable=False)
    st.caption(RETRAIN_DISCLAIMER)

    st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)
    st.subheader("🎯 Definisi target & aturan keputusan")
    tt1, tt2, tt3 = st.columns(3)
    tt1.metric("Syarat awal", "/".join(sorted(ta_meta.get("starting_regimes", []))) or "-")
    tt2.metric("Target regime", "/".join(ta_meta.get("target_regimes", [])) or "-")
    tt3.metric("Horizon", f"{ta_meta.get('horizon_trading_days', '-')} hari trading")
    st.markdown(
        f"""
        - Hanya menyekor ticker yang **SAAT INI** berada di regime {"/".join(sorted(ta_meta.get("starting_regimes", [])))}
          -- ticker di regime lain tidak dinilai model ini sama sekali (di luar apa yang dipelajari saat training).
        - **POTENSIAL** — probabilitas ≥ {ta_threshold*100:.0f}% untuk berpindah ke
          {"/".join(ta_meta.get("target_regimes", []))} dan **bertahan** di sana (tidak jatuh lagi ke
          {"/".join(sorted(ta_meta.get("starting_regimes", [])))} atau overextended) selama
          ≥{ta_meta.get("hold_trading_days", "-")} hari perdagangan, dalam {ta_meta.get("horizon_trading_days", "-")} hari ke depan.
        - **BELUM** — probabilitas di bawah threshold itu.
        """
    )

    st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)
    render_features_and_tickers(ta_meta)
