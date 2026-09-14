import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st

from app.auth import require_login
from app.data import load_ihsg_trend
from app.positions import close_position, load_positions_with_progress
from app.style import inject_base_css, render_developer_footer, render_ihsg_context

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
    "hari dan memberi peringatan dini kalau harga terlihat mulai mengarah ke stop-loss, SEBELUM "
    "benar-benar menyentuh angka itu. Tandai satu saham dari halaman **Detail Saham** saat "
    "keputusannya BUY."
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
        n_warning = int(positions["warning"].sum())
        c1, c2 = st.columns(2)
        c1.metric("Posisi aktif", len(positions))
        c2.metric("⚠️ Perlu perhatian", n_warning, delta_color="inverse" if n_warning else "off")

        for _, pos in positions.sort_values("warning", ascending=False).iterrows():
            with st.container():
                st.markdown('<div class="mystocks-card">', unsafe_allow_html=True)
                h1, h2, h3, h4 = st.columns([1.5, 1, 1, 1])
                with h1:
                    st.markdown(f"<div class='mystocks-ticker'>{pos['stock_code']}</div>", unsafe_allow_html=True)
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

                if pos["warning"]:
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
