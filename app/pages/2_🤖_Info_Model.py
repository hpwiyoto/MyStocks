import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st

from app.data import days_since, feature_daily_row_count, load_model_metadata
from app.style import inject_base_css, render_developer_footer

st.set_page_config(page_title="MyStocks — Info Model", page_icon="🤖", layout="wide")
inject_base_css()
render_developer_footer()

if st.button("← Kembali ke Home"):
    st.switch_page("Home.py")

st.title("🤖 Info Model")
st.caption("Model yang sedang dipakai untuk menghasilkan prediksi di halaman Screener.")

meta = load_model_metadata()
current_rows = feature_daily_row_count()
trained_rows = meta["n_training_rows"]
new_rows = current_rows - trained_rows
age_days = days_since(meta["trained_at"])

c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    st.markdown("<span class='mystocks-muted'>Model</span>", unsafe_allow_html=True)
    st.markdown(f"<div class='mystocks-metric-value'>{meta['model_version']}</div>", unsafe_allow_html=True)
c2.metric("Dilatih pada", meta["trained_at"], delta=f"{age_days} hari lalu", delta_color="off")
c3.metric("Baseline win rate (base_rate)", f"{meta['base_rate']*100:.1f}%")

st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)

st.subheader("📈 Pengingat retrain")
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

st.caption(
    "Retrain sengaja tidak disediakan sebagai tombol di aplikasi ini — proses training butuh review "
    "manusia atas walk-forward validation & SHAP sebelum model baru dipercaya (lihat docs/BUILD_PROMPTS.md "
    "Fase 3). Halaman ini hanya menampilkan indikator, bukan memicu training."
)

st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)

st.subheader("🎯 Definisi target & aturan keputusan")
t1, t2, t3 = st.columns(3)
t1.metric("Target profit", f"{meta['target_pct']*100:.1f}%")
t2.metric("Stop loss", f"{meta['stop_pct']*100:.1f}%")
t3.metric("Horizon", f"{meta['horizon_days']} hari trading")

st.markdown(
    """
    - **BUY** — probabilitas ≥ 50%
    - **WATCH** — probabilitas ≥ base rate historis model, tapi < 50%
    - **AVOID** — probabilitas di bawah base rate (tidak ada edge)
    """
)

st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)

st.subheader(f"🧬 Fitur yang dipakai model ({len(meta['feature_cols'])})")
st.dataframe(
    {"feature": meta["feature_cols"]},
    width="stretch",
    hide_index=True,
    height=300,
)

st.subheader("🏢 Ticker dalam data training")
st.write(", ".join(meta.get("tickers") or []))
