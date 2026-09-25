import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st

from app.auth import require_login
from app.data import best_swing_config_id, days_since, feature_daily_row_count, load_ihsg_trend, load_model_metadata
from app.style import SUSPENSION_RISK_NOTE, inject_base_css, render_developer_footer, render_ihsg_context
from engine.swing_configs import CONFIG_BY_ID, DEFAULT_CONFIG_ID, SWING_CONFIGS

st.set_page_config(page_title="MyStocks — Info Model", page_icon="🤖", layout="wide")
inject_base_css()
require_login("Info Model")
render_developer_footer()

if st.button("← Kembali ke Home"):
    st.switch_page("Home.py")

st.title("🤖 Info Model")
render_ihsg_context(load_ihsg_trend())
st.caption("Detail model machine learning yang dipakai aplikasi ini -- Swing.")


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
st.header("🎯 Model Swing")

# Same shared `persist_swing_config_id` session_state slot the Swing page's
# config toggle (engine.swing_configs) writes -- picking a config here
# updates it everywhere else too (Home/Swing/Detail Saham/Rekomendasi
# Emitten), and vice versa, instead of this page silently locking to the
# default while the rest of the app shows something else.
_best_id = best_swing_config_id()


def _display_label(c: dict) -> str:
    return ("⭐ " if c["id"] == _best_id else "") + c["label"]


_config_display = [_display_label(c) for c in SWING_CONFIGS]
_persisted_config_id = st.session_state.get("persist_swing_config_id", DEFAULT_CONFIG_ID)
_persisted_display = _display_label(CONFIG_BY_ID.get(_persisted_config_id, CONFIG_BY_ID[DEFAULT_CONFIG_ID]))

# key= + on_change, not index= recomputed every run -- see app/pages/
# 2_Swing.py's identical toggle for why: recomputing `index` from a plain
# session_state var that's only written further down the SAME run made
# the widget lag one run behind the actual click (needed two clicks to
# register). `key=` lets Streamlit own the value directly; only seeded
# from the persisted slot when the widget doesn't have one yet (i.e.
# right after a page switch, which clears this key but not `persist_
# swing_config_id`), so an in-progress click is never stomped.
_INFO_MODEL_CONFIG_WIDGET_KEY = "info_model_config_widget"


def _on_info_model_config_change():
    picked = st.session_state[_INFO_MODEL_CONFIG_WIDGET_KEY]
    st.session_state["persist_swing_config_id"] = SWING_CONFIGS[_config_display.index(picked)]["id"]


if _INFO_MODEL_CONFIG_WIDGET_KEY not in st.session_state:
    st.session_state[_INFO_MODEL_CONFIG_WIDGET_KEY] = _persisted_display

# segmented_control, not selectbox -- direct user request for all 5
# options visible horizontally instead of hidden behind a dropdown.
st.segmented_control(
    "🎯 Konfigurasi target Swing", _config_display,
    selection_mode="single",
    required=True,
    key=_INFO_MODEL_CONFIG_WIDGET_KEY,
    on_change=_on_info_model_config_change,
    help="⭐ = precision/Wilson LB pooled TERTINGGI di antara ke-5 konfigurasi -- bisa saja BUKAN "
         "konfigurasi #1 (yang diurutkan berdasarkan lift atas baseline acaknya sendiri, metrik "
         "berbeda -- lihat tabel perbandingan di bawah).",
)
_selected_display = st.session_state[_INFO_MODEL_CONFIG_WIDGET_KEY]
selected_config = SWING_CONFIGS[_config_display.index(_selected_display)]
st.session_state["persist_swing_config_id"] = selected_config["id"]

meta = load_model_metadata(selected_config["model_version"])
st.caption(f"Konfigurasi #{selected_config['rank']} dari 20 (lift top-5% {selected_config['top5_lift']:+.3f} atas baseline acak).")
render_header(meta, "Baseline win rate (base_rate)")
st.caption(
    f"Ini angka **pembanding acak** (kalau asal pilih saham tanpa strategi apa pun, kira-kira segini "
    f"sering yang naik ≥{meta['target_pct']*100:.0f}% sebelum -{meta['stop_pct']*100:.1f}% dalam "
    f"{meta['horizon_days']} hari) -- bukan skor performa model. Model dianggap bekerja kalau bisa "
    "MENGGESER probabilitas jauh dari angka ini: BUY seharusnya jauh di atas, AVOID jauh di bawah. "
    "Lihat halaman **Swing** untuk win rate sesungguhnya per keputusan."
)

with st.expander("📊 Bandingkan semua 5 konfigurasi"):
    _cmp_rows = []
    for c in SWING_CONFIGS:
        _cm = load_model_metadata(c["model_version"])
        _cwf = _cm.get("walk_forward_validation") or {}
        _cp = _cwf.get("pooled") or {}
        _cpf = _cm.get("profit_factor_analysis") or {}
        _cmp_rows.append({
            "Konfigurasi": _display_label(c), "Peringkat": c["rank"], "Lift top-5%": f"{c['top5_lift']:+.3f}",
            "Threshold BUY": f"{_cwf.get('buy_threshold', 0)*100:.0f}%",
            "Precision (pooled)": f"{_cp.get('precision', 0)*100:.1f}%" if _cp else "-",
            "Wilson LB 95%": f"{_cp.get('wilson_lb_95', 0)*100:.1f}%" if _cp else "-",
            "n sinyal BUY": _cp.get("n_trades", "-"),
            "Profit Factor": f"{_cpf['profit_factor_realistic']:.2f}" if _cpf else "-",
            "Expectancy/hari": f"{_cpf['expectancy_per_day_pct']:+.2f}%" if _cpf else "-",
        })
    # Sorted by expectancy_per_day_pct DESC (not by rank/Wilson LB) --
    # deliberate: this is the decision-relevant ordering per docs/
    # RESEARCH_LOG.md's finding below, distinct from every other column's
    # own ranking in this table.
    _cmp_rows.sort(
        key=lambda r: float(r["Expectancy/hari"].rstrip("%")) if r["Expectancy/hari"] != "-" else float("-inf"),
        reverse=True,
    )
    st.dataframe(_cmp_rows, width="stretch", hide_index=True)
    st.caption(
        "⭐ = Wilson LB pooled tertinggi (angka praktis \"kalau saya ikuti tiap sinyal BUY-nya, berapa "
        "peluang menang\"), bukan berarti peringkat #1. Semua angka di sini dihitung pooled (trade-"
        "weighted) per konfigurasi masing-masing, bukan dibandingkan pada threshold yang sama -- tiap "
        "konfigurasi punya threshold BUY-nya sendiri hasil "
        "tuning terpisah. Lihat `scripts/search_swing_target.py` untuk metodologi pencarian 20 "
        "konfigurasinya, dan `scripts/train_v5_variants.py` untuk cara 4 konfigurasi selain #1 dilatih."
    )
    st.info(
        "💡 **Profit Factor dan Wilson LB/Precision menghasilkan urutan yang SAMA** untuk ke-5 konfigurasi "
        "ini (semuanya pakai rasio target:stop 2:1) -- profit factor sendiri tidak menambah informasi baru "
        "di sini. Tapi **Expectancy/hari** (profit factor digabung frekuensi sinyal & horizon, tabel di atas "
        "diurutkan berdasarkan ini) justru MEMBALIK urutannya: konfigurasi **default (10%/-5%/5 hari)** yang "
        "sedang berjalan menang di kecepatan profit (+2,45%/hari) meski win-rate & profit factor-nya PALING "
        "RENDAH dari kelima config -- target besar dicapai di horizon tercepat + frekuensi sinyal tertinggi. "
        "Detail metodologi (return realistis dengan gap harga open, bukan cuma target/stop nominal) di "
        "`scripts/compute_profit_factor.py` dan `docs/RESEARCH_LOG.md`.",
        icon="💡",
    )

swing_wf = meta.get("walk_forward_validation") or {}
swing_ml = swing_wf.get("avg_ml_metrics") or {}
swing_pooled = swing_wf.get("pooled") or {}
swing_wf_threshold = swing_wf.get("buy_threshold")
sp1, sp2, sp3, sp4 = st.columns(4)
sp1.metric(
    f"Precision @ threshold {swing_wf_threshold*100:.0f}% (pooled, walk-forward)" if swing_wf_threshold else "Precision (pooled, walk-forward)",
    f"{swing_pooled.get('precision', swing_ml.get('precision', 0))*100:.1f}%" if (swing_pooled or swing_ml) else "-",
    help="Dihitung per-transaksi di seluruh fold (bukan rata-rata sederhana antar fold) -- lihat catatan di bawah.",
)
sp2.metric(
    "Batas bawah keyakinan 95% (Wilson LB)",
    f"{swing_pooled['wilson_lb_95']*100:.1f}%" if swing_pooled else "-",
    help="Perkiraan konservatif dari precision di atas -- lebih aman dipakai untuk ekspektasi daripada titik estimasi saja.",
)
sp3.metric("ROC-AUC (walk-forward)", f"{swing_ml.get('roc_auc', 0):.3f}" if swing_ml else "-")
sp4.metric("BUY threshold saat ini (live)", f"{swing_wf_threshold*100:.0f}%" if swing_wf_threshold else "-")

# Profit factor / expectancy: separate row, separate data source
# (profit_factor_analysis, from scripts/compute_profit_factor.py) --
# genuinely different question from precision/Wilson-LB above ("seberapa
# sering menang" vs "seberapa banyak duit dihasilkan per hari"). See the
# comparison-table caption above for why the two can disagree.
swing_pf = meta.get("profit_factor_analysis") or {}
if swing_pf:
    pf1, pf2, pf3 = st.columns(3)
    pf1.metric(
        "Profit Factor (realistis, gap-adjusted)", f"{swing_pf['profit_factor_realistic']:.2f}",
        help="Total untung / total rugi dari sinyal BUY yang diambil, memakai harga OPEN di hari exit "
             "(bukan cuma target/stop nominal) supaya gap harga semalam ikut terhitung.",
    )
    pf2.metric(
        "Expectancy per transaksi", f"{swing_pf['expectancy_realistic_pct']:+.2f}%",
        help="Rata-rata untung/rugi per sinyal BUY yang diambil (realistis, gap-adjusted).",
    )
    pf3.metric(
        "Expectancy per hari (kecepatan profit)", f"{swing_pf['expectancy_per_day_pct']:+.2f}%",
        help="Expectancy per transaksi dibagi horizon hari -- angka yang dipakai untuk memilih config "
             "default, bukan precision/win-rate. Lihat kotak info di 'Bandingkan semua 5 konfigurasi' di atas.",
    )
    st.caption(
        f"Dihitung dari {swing_pf['n_trades']} sinyal BUY ({swing_pf['signals_per_day']:.2f} sinyal/hari, "
        f"win rate {swing_pf['win_rate']*100:.1f}%) -- sumber: `scripts/compute_profit_factor.py`, "
        "penjelasan lengkap di `docs/RESEARCH_LOG.md`."
    )

if swing_pooled:
    st.caption(
        f"✅ **Angka precision di atas sudah di-pooling per-transaksi** ({swing_pooled['n_trades']} sinyal BUY, "
        f"{swing_pooled['wins']} menang, di seluruh fold walk-forward digabung) -- bukan sekadar rata-rata "
        f"precision antar fold. Bedanya nyata: rata-rata sederhana antar fold sempat menunjukkan **"
        f"{swing_ml.get('precision', 0)*100:.1f}%**, lebih tinggi dari angka pooled **"
        f"{swing_pooled['precision']*100:.1f}%** -- selisih itu murni artefak penghitungan (salah satu dari 5 "
        "fold walk-forward hampir selalu punya ~0 sinyal BUY, tapi ikut dihitung sama beratnya dengan fold "
        "yang punya ratusan transaksi sungguhan). Ditemukan lewat `scripts/tune_v5_extended.py` (pencarian "
        "40 konfigurasi hyperparameter -- kandidat yang tampak menang di rata-rata-sederhana ternyata KALAH "
        "begitu dicek pooled, termasuk konfigurasi yang SAAT INI dipakai model ini). Angka pooled di atas "
        "yang seharusnya dipercaya untuk pertanyaan 'kalau saya ikuti tiap sinyal BUY, berapa peluang menang'."
    )
if selected_config["id"] == DEFAULT_CONFIG_ID:
    st.caption(
        "⚠️ **Angka precision di atas adalah rata-rata 4 fold walk-forward, bukan angka tunggal yang "
        "stabil.** Dicek per-fold secara terpisah (`scripts/check_fold_drift.py`): performanya bervariasi "
        "72-88% antar fold, dan fold PALING BARU justru yang PALING LEMAH (~71-72%, sinyal BUY paling "
        "jarang muncul) -- berkorelasi dengan periode IHSG sedang turun. Sudah dicoba diperbaiki dengan "
        "menambahkan fitur tren IHSG, tapi terbukti memperparah drastis (`scripts/test_ihsg_regime_feature.py`), "
        "jadi belum ada perbaikan yang diterapkan. Model ini perlu dipantau berkala, bukan dianggap "
        "'sudah pasti bagus selamanya' hanya dari validasi sekali di tanggal training."
    )
    st.caption(
        "🔎 **Pengingat pemantauan**: analisis drift terakhir dijalankan **7 September 2026** (untuk "
        "target lama 5%/-2,5%/10 hari -- belum diulang untuk target baru 10%/-5%/5 hari). Disarankan "
        "jalankan `python -m scripts.check_fold_drift` lagi setiap 2-3 bulan, atau lebih cepat kalau IHSG "
        "baru saja bergerak besar (naik/turun >10% dalam sebulan) -- bukan proses otomatis, perlu dijalankan manual."
    )
else:
    st.caption(
        "🔎 Konfigurasi ini adalah salah satu dari 4 runner-up (`scripts/train_v5_variants.py`) -- belum "
        "punya analisis drift per-fold terpisah (`scripts/check_fold_drift.py` baru pernah dijalankan "
        "untuk konfigurasi default #1). Anggap validasi walk-forward di atas sebagai gambaran awal, bukan "
        "pemantauan berkelanjutan."
    )
st.caption(
    "**Kenapa satu saham WATCH bisa menampilkan probabilitas serendah 30%an di Detail Saham**: "
    "probabilitas mentah dari model itu terus-menerus (0-100%), bukan skor keyakinan model pada "
    "dirinya sendiri. Angka precision di atas HANYA berlaku untuk saham yang probabilitasnya sudah "
    "melewati ambang BUY -- bukan klaim bahwa SETIAP angka yang ditampilkan seharusnya tinggi. "
    "Base rate historis (tanpa strategi apa pun) sekitar 30%, jadi probabilitas 30%an persis ada di "
    "wilayah WATCH (di atas base rate, belum cukup tinggi untuk BUY) -- itu model bekerja sesuai "
    "rancangannya, bukan tanda model gagal atau tidak akurat."
)
st.caption(SUSPENSION_RISK_NOTE)

st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)
render_retrain_reminder(meta, current_rows)
st.caption(RETRAIN_DISCLAIMER)

st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)
st.subheader("🎯 Definisi target & aturan keputusan")
t1, t2, t3 = st.columns(3)
t1.metric("Target profit", f"{meta['target_pct']*100:.1f}%")
t2.metric("Stop loss", f"{meta['stop_pct']*100:.1f}%")
t3.metric("Horizon", f"{meta['horizon_days']} hari trading")
_ihsg_trend_for_rule = load_ihsg_trend()
_ihsg_declining_now = bool(_ihsg_trend_for_rule and _ihsg_trend_for_rule.get("ret_20d") is not None and _ihsg_trend_for_rule["ret_20d"] < 0)
# Per-config threshold, NOT engine.decision's module constants directly --
# each of the 5 configs has its own metadata-stored buy_threshold, and
# engine.predict.run() applies a uniform "+0.05 during IHSG decline" bump
# to WHICHEVER threshold a config actually uses (see that function's
# docstring) -- reproduced here so this page always shows the config
# actually selected above, not always the default's numbers.
_model_buy_threshold = swing_wf_threshold or 0.60
_model_ihsg_decline_threshold = round(min(_model_buy_threshold + 0.05, 0.95), 2)
_active_buy_threshold = _model_ihsg_decline_threshold if _ihsg_declining_now else _model_buy_threshold
_ihsg_decline_note = (
    "Terbukti lewat backtest (`scripts/test_regime_conditional_threshold.py`) untuk konfigurasi default: "
    "sinyal BUY saat IHSG turun historisnya menang lebih jarang (71,6% vs 75,2%)."
    if selected_config["id"] == DEFAULT_CONFIG_ID else
    "Bump +5pp ini warisan dari konfigurasi default, BELUM divalidasi ulang secara terpisah untuk "
    "konfigurasi ini (`scripts/test_regime_conditional_threshold.py` belum dijalankan ulang per-config)."
)
st.markdown(
    f"""
    - **BUY** — probabilitas ≥ {_model_buy_threshold*100:.0f}% -- **kecuali saat IHSG sendiri sedang turun**
      (return 20 hari negatif), di mana ambangnya naik ke {_model_ihsg_decline_threshold*100:.0f}%. {_ihsg_decline_note}
      **Ambang yang berlaku hari ini: {_active_buy_threshold*100:.0f}%**
      ({"IHSG sedang turun" if _ihsg_declining_now else "IHSG tidak sedang turun"}).
    - **WATCH** — probabilitas ≥ base rate historis model, tapi < ambang BUY yang berlaku hari itu
    - **AVOID** — probabilitas di bawah base rate (tidak ada edge), atau harga saham ≤ Rp50 (floor gocap,
      lihat `engine.decision.GOCAP_PRICE_FLOOR` — dipaksa AVOID apa pun probabilitasnya)
    """
)

st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)
render_features_and_tickers(meta)
