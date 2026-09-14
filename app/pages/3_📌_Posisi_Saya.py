import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import streamlit as st

from app.auth import require_login
from app.data import load_ihsg_trend
from app.positions import STATUS_LABELS, close_position, load_positions_with_progress
from app.style import inject_base_css, position_status_badge, render_developer_footer, render_ihsg_context

# Display order: most urgent first -- a stock already past target/stop
# (should be closed NOW) outranks an early warning, which outranks a
# merely-expired prediction window, etc. Keys are app.positions.
# STATUS_LABELS' keys; see that module's docstring for what each means.
STATUS_SORT_ORDER = {"target_hit": 0, "stop_hit": 0, "warning": 1, "expired": 2, "under_pressure": 3, "on_track": 4}

st.set_page_config(page_title="MyStocks — Posisi Saya", page_icon="📌", layout="wide")
inject_base_css()
require_login("Posisi Saya")
render_developer_footer()

if st.button("← Kembali ke Home"):
    st.switch_page("Home.py")

st.title("📌 Posisi Saya")
render_ihsg_context(load_ihsg_trend())
st.caption(
    "Saham yang Anda tandai sebagai sudah dibeli -- sistem terus memantau pergerakannya setiap "
    "hari, menandai statusnya (✅ On Track / 🟡 Dalam Tekanan / ⚠️ Peringatan Dini / ⏳ Kadaluarsa / "
    "🎯🛑 sudah kena target atau stop), dan memberi peringatan dini kalau harga terlihat mulai "
    "mengarah ke stop-loss, SEBELUM benar-benar menyentuh angka itu. Tandai satu saham dari "
    "halaman **Detail Saham** saat keputusannya BUY."
)
st.caption(
    "⚠️ **Batasan yang jujur perlu diketahui** (`scripts/test_early_warning_rule.py`): dari histori, "
    "peringatan dini ini menangkap ~5% dari kerugian yang akhirnya kena stop-loss (rata-rata "
    "menyelamatkan ~3 poin persentase kerugian kalau tertangkap) dengan alarm palsu yang sangat "
    "jarang (0,2%) -- TAPI 69% dari histori kerugian ternyata kena stop-loss di HARI PERTAMA "
    "setelah beli, jadi tidak ada hari sebelumnya untuk memberi peringatan sama sekali. Ini bantuan "
    "nyata tapi terbatas, bukan jaminan menangkap semua kerugian."
)

user_email = st.user.email
tab_active, tab_closed = st.tabs(["🟢 Aktif", "📁 Riwayat (ditutup)"])

with tab_active:
    with st.spinner("Memuat posisi aktif..."):
        positions = load_positions_with_progress(user_email, status="active")

    if positions.empty:
        st.info(
            "Belum ada posisi yang ditandai. Buka halaman **Detail Saham** untuk saham dengan "
            "keputusan BUY, lalu klik \"📌 Tandai Saya Beli Ini\"."
        )
    else:
        n_warning = int((positions["status"] == "warning").sum())
        n_action = int(positions["status"].isin(["target_hit", "stop_hit"]).sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Posisi aktif", len(positions))
        c2.metric("⚠️ Peringatan dini", n_warning, delta_color="inverse" if n_warning else "off")
        c3.metric("🎯 Siap ditutup", n_action, delta_color="inverse" if n_action else "off")

        positions["_sort_key"] = positions["status"].map(STATUS_SORT_ORDER).fillna(5)
        for _, pos in positions.sort_values("_sort_key").iterrows():
            with st.container():
                st.markdown('<div class="mystocks-card">', unsafe_allow_html=True)
                h1, h2, h3, h4 = st.columns([1.5, 1, 1, 1])
                with h1:
                    status_badge = position_status_badge(pos["status"], STATUS_LABELS.get(pos["status"], pos["status"]))
                    st.markdown(
                        f"<div class='mystocks-ticker'>{pos['stock_code']}</div>"
                        f"<div style='margin:0.3rem 0;'>{status_badge}</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Beli {pos['entry_date']} @ {pos['entry_price']:,.0f} -- {pos['days_held']} hari lalu")
                with h2:
                    price_txt = f"{pos['current_price']:,.0f}" if pos["current_price"] is not None else "-"
                    st.metric("Harga sekarang", price_txt, delta=f"{pos['pct_change_from_entry']:+.1f}%" if pos["pct_change_from_entry"] is not None else None)
                with h3:
                    st.caption("Stop Loss")
                    st.write(f"{pos['stop_loss_price']:,.0f}")
                    if pos["pct_to_stop"] is not None:
                        st.caption(f"jarak {pos['pct_to_stop']:.1f}%")
                with h4:
                    st.caption("Take Profit")
                    st.write(f"{pos['take_profit_price']:,.0f}")
                    if pos["pct_to_target"] is not None:
                        st.caption(f"jarak {pos['pct_to_target']:.1f}%")

                # Progress hari: divisualkan sebagai progress bar terhadap
                # horizon model (bukan cuma angka) -- direct user request
                # ("visualkan progress sudah berapa hari"). horizon_days
                # dari snapshot entry_horizon_days kalau ada (lihat
                # app.positions.load_positions_with_progress), fallback ke
                # None untuk posisi lama sebelum kolom ini ada.
                _horizon = pos.get("entry_horizon_days")
                if pos["days_held"] is not None and pd.notna(_horizon) and _horizon > 0:
                    _frac = min(pos["days_held"] / _horizon, 1.0)
                    st.progress(_frac, text=f"Hari ke-{pos['days_held']} dari perkiraan {int(_horizon)} hari (horizon model saat ditandai)")

                # Awal vs Sekarang -- direct user request ("direcord hasil
                # rekomendasi swing nya apa saat di klik tandai beli...
                # dan visualkan current statusnya"): what the recommendation
                # said at mark-time, compared to what it says right now.
                def _fmt_pct(v):
                    return f"{float(v)*100:.1f}%" if v is not None and v == v else "-"

                def _fmt_regime(v):
                    return v.replace("_", " ") if isinstance(v, str) else "-"

                cmp1, cmp2 = st.columns(2)
                with cmp1:
                    st.caption("📸 Saat Ditandai")
                    if pd.notna(pos.get("entry_target_pct")) and pd.notna(pos.get("entry_horizon_days")):
                        st.write(f"Konfigurasi: {_fmt_pct(pos['entry_target_pct'])} / -{_fmt_pct(pos['entry_stop_pct'])} / {int(pos['entry_horizon_days'])} hari")
                    else:
                        st.write("Konfigurasi: - (ditandai sebelum fitur ini ada)")
                    st.write(f"Probabilitas: {_fmt_pct(pos.get('entry_probability'))}")
                    st.write(f"Regime: {_fmt_regime(pos.get('entry_regime'))}")
                    st.write(f"Fase Wyckoff: {_fmt_regime(pos.get('entry_wyckoff_phase'))}")
                with cmp2:
                    st.caption("📡 Sekarang")
                    st.write(f"Probabilitas: {_fmt_pct(pos.get('current_probability'))}"
                              + (f" ({pos['current_decision']})" if pos.get("current_decision") else ""))
                    st.write(f"Regime: {_fmt_regime(pos.get('current_regime'))}")
                    st.write(f"Fase Wyckoff: {_fmt_regime(pos.get('current_wyckoff_phase'))}")

                if pos["status"] == "warning":
                    detail = pos["warning_detail"]
                    fired = [name for name, hit in {
                        "harga di bawah EMA9": detail["signals"]["trend_broken"],
                        "momentum RSI melemah": detail["signals"]["momentum_weakening"],
                        "aliran dana keluar (CMF negatif)": detail["signals"]["money_flowing_out"],
                    }.items() if hit]
                    st.error(
                        f"⚠️ **Peringatan dini**: posisi sedang rugi DAN {len(fired)} dari 3 tanda pelemahan "
                        f"aktif ({', '.join(fired)}) -- pertimbangkan keluar sekarang daripada menunggu "
                        "stop-loss penuh. Bukan jaminan, lihat catatan batasan di atas.",
                        icon="⚠️",
                    )
                elif pos["status"] == "target_hit":
                    st.success("🎯 Harga sudah menyentuh/melewati Take Profit -- pertimbangkan tutup posisi ini.", icon="🎯")
                elif pos["status"] == "stop_hit":
                    st.error("🛑 Harga sudah menyentuh/melewati Stop Loss -- pertimbangkan tutup posisi ini.", icon="🛑")
                elif pos["status"] == "expired":
                    st.warning(
                        "⏳ Sudah melewati perkiraan jendela waktu prediksi model untuk saham ini -- "
                        "prediksi awal (target/stop dalam horizon tertentu) sudah tidak berlaku lagi, "
                        "posisi ini sekarang di luar apa yang diperkirakan model. Evaluasi ulang manual.",
                        icon="⏳",
                    )

                b1, b2, b3 = st.columns(3)
                if b1.button("✅ Tutup: Kena Target", key=f"target_{pos['id']}", width="stretch"):
                    close_position(int(pos["id"]), float(pos["current_price"] or pos["take_profit_price"]), "target_hit")
                    st.rerun()
                if b2.button("🛑 Tutup: Kena Stop", key=f"stop_{pos['id']}", width="stretch"):
                    close_position(int(pos["id"]), float(pos["current_price"] or pos["stop_loss_price"]), "stop_hit")
                    st.rerun()
                if b3.button("🚪 Tutup Manual", key=f"manual_{pos['id']}", width="stretch"):
                    close_position(int(pos["id"]), float(pos["current_price"] or pos["entry_price"]), "manual")
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

with tab_closed:
    with st.spinner("Memuat riwayat..."):
        closed = load_positions_with_progress(user_email, status="closed")
    if closed.empty:
        st.info("Belum ada posisi yang ditutup.")
    else:
        closed = closed.copy()
        closed["hasil_pct"] = (closed["closed_price"].astype(float) - closed["entry_price"].astype(float)) / closed["entry_price"].astype(float) * 100
        reason_labels = {"target_hit": "🎯 Kena Target", "stop_hit": "🛑 Kena Stop", "manual": "🚪 Manual", "early_warning": "⚠️ Peringatan Dini"}
        closed["alasan"] = closed["closed_reason"].map(reason_labels).fillna(closed["closed_reason"])
        table = closed[["stock_code", "entry_date", "entry_price", "closed_date", "closed_price", "hasil_pct", "alasan"]]
        st.dataframe(
            table, width="stretch", hide_index=True,
            column_config={
                "stock_code": st.column_config.TextColumn("Kode"),
                "entry_date": st.column_config.TextColumn("Tgl Beli"),
                "entry_price": st.column_config.NumberColumn("Harga Beli", format="%.0f"),
                "closed_date": st.column_config.TextColumn("Tgl Tutup"),
                "closed_price": st.column_config.NumberColumn("Harga Tutup", format="%.0f"),
                "hasil_pct": st.column_config.NumberColumn("Hasil", format="%+.1f%%"),
                "alasan": st.column_config.TextColumn("Alasan"),
            },
        )
        win_rate = (closed["hasil_pct"] > 0).mean() * 100
        st.caption(f"Win rate posisi yang sudah ditutup: {win_rate:.0f}% dari {len(closed)} posisi.")
