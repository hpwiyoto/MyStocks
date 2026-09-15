# Research Log — Swing/Momentum/Turnaround (konsolidasi lintas-sesi)

Dokumen ini menggabungkan riset dari **banyak sesi Claude Code yang berjalan
terpisah** (2026-09-08 sampai 2026-09-15, 40 commit) menjadi satu bundel
per-tema, supaya temuan yang saling terkait tidak tercecer di percakapan
yang berbeda-beda. Sumber kebenarannya tetap `git log` (setiap commit punya
penjelasan "kenapa" + hasil terukur di message-nya) -- dokumen ini adalah
ringkasan yang dikorelasikan, bukan pengganti riwayat commit itu sendiri.

**Cara pakai**: kalau Anda lupa "kita sudah coba X belum", cari di sini
dulu berdasarkan tema. Tiap butir menyebut commit hash pendek -- jalankan
`git show <hash>` untuk detail lengkap kapan saja.

**Konvensi status**: ✅ Diadopsi/live sekarang · ❌ Ditolak (diuji, tidak
dipakai, tetap didokumentasikan sebagai hasil negatif permanen) · 📊
Riset murni (belum perlu keputusan ship/tidak) · 🗑️ Dihapus dari app
(kode tetap ada di git, hanya tidak dipakai lagi).

---

## 📍 Status Saat Ini (ringkasan, per 2026-09-15)

**Model Swing** -- sekarang punya **5 konfigurasi target/stop/horizon**
yang bisa ditoggle di UI (dulu cuma 1: 5%/-2.5%/10 hari). Semua pakai 44
fitur yang sama, threshold di-tune independen per konfigurasi:

| Config | Target/Stop/Horizon | Threshold BUY | Precision (pooled) | Wilson LB 95% | n trades |
|---|---|---|---|---|---|
| **t10_h5 (default)** | 10% / -5% / 5 hari | 60% | 74,4% | 71,3% | 804 |
| t7_h5 | 7% / -3,5% / 5 hari | 70% | 81,1% | 77,7% | 586 |
| t10_h10 | 10% / -5% / 10 hari | 65% | 75,7% | 72,5% | 750 |
| t7_h10 | 7% / -3,5% / 10 hari | 70% | 81,4% | **78,1%** | 617 |
| t5_h5 | 5% / -2,5% / 5 hari | 70% | 81,9% | 78,2% | 496 |

Default dipilih bukan karena precision tertinggi (t5_h5/t7_h10 lebih
tinggi) tapi karena `top5_lift` (metrik yang adil lintas-target) dan
kecepatan modal (profit factor × frekuensi sinyal × horizon) -- lihat
Thread A.

**Turnaround** -- 🗑️ **dipensiunkan dari app** (halaman dihapus, tidak
lagi di pipeline harian). Kode/model/skrip riset tetap ada di git, hanya
tidak aktif. Alasan: baik Turnaround asli maupun kandidat penggantinya
("Turnaround v2") tidak pernah tembus 60% win rate, sementara Swing top-2
mencapai 64-69%. Lihat Thread C.

**Rekomendasi Emitten** -- sekarang gabungan **2 alat** (Swing + Momentum),
bukan 3 -- kolom Turnaround dihapus seiring pensiunnya Turnaround.

**Momentum Screener** -- tidak berubah sejak dokumen README terakhir:
bottoming + momentum menguat + CMF<0 + RVOL≥0,8 + AVWAP, win rate 42,4%
(Wilson LB 38,3%, n=523). Banyak kandidat kriteria baru diuji minggu ini,
**semua ditolak** -- lihat Thread B.

**Fitur baru di app** (di luar model): Posisi Saya (tracking saham yang
ditandai beli + early-warning), badge status Wyckoff (display-only, bukan
fitur model), badge "Keyakinan Tinggi" + peringatan saham overextended,
filter likuiditas opsional, exclude saham suspend dari semua screener,
chart IHSG di Home. Lihat Thread D & E.

---

## Thread A — Swing: definisi target & tuning (yang paling besar dampaknya)

Urutan logis (bukan sekadar kronologis) dari akar masalah ke solusi final:

1. **`d8b245d`** 📊 — Dicek: apakah target 5%/-2,5%/10 hari itu sendiri
   optimal? TIDAK. Grid search 20 kombinasi target/stop/horizon (rasio
   2:1 ditahan tetap) pakai metrik `top5_lift` (adil lintas-target).
   Pemenang: **10%/-5%/5 hari** (lift +0,228) vs yang lama rank ke-8
   (lift +0,168). Pola jelas: horizon lebih pendek menang di semua
   ukuran target.
2. **`c786154`** ✅ — Diadopsi atas persetujuan eksplisit user ("saya
   ingin ke hasil yang paling optimal"). Retrain + threshold baru
   (60%, precision 71,9%).
3. **`149d2f0`** — Sinkronisasi semua teks UI + angka turunan (badge
   overextended, catatan risiko suspensi) ke target baru.
4. **`c50531a` → `2aba98a` → `161539f` → `83cbe57`** ✅ — User minta 4
   runner-up juga bisa ditoggle, bukan cuma pemenang tunggal. Dibangun
   sebagai model saudara independen (`engine/swing_configs.py`), lalu
   dipoles: bintang penanda Wilson-LB-terbaik, perbaikan bug harus
   klik 2x, ganti dropdown jadi tombol horizontal.
   - Bug nyata ditemukan di sini: `decide()` awalnya selalu pakai
     threshold config DEFAULT untuk SEMUA model, jadi 4 model varian
     salah diklasifikasi BUY/WATCH sampai diperbaiki.
5. **`2535081` → `31e85d1` → `89c3d3d`** ✅ — Hyperparameter XGBoost
   ternyata belum pernah di-tune ulang untuk target baru (masih warisan
   target lama). Ditemukan perbaikan nyata: `max_depth` 3→4, `eta`
   0,05→0,03 untuk config default (+2,63pp Wilson LB) dan t10_h10
   (+2,19pp). t7_h5/t7_h10/t5_h5 awalnya HANYA "lead belum terverifikasi"
   (magnitude terlalu kecil, mirip pola noise yang pernah ketemu di
   riset window/MA di bawah).
6. **`965bd56` → `ead138b`** ✅ — Verifikasi fine-sweep: t7_h10 punya
   peak bersih di eta=0,03 (real finding, diadopsi pagi ini). t5_h5
   TIDAK punya peak bersih (noise, tetap pakai hyperparameter lama).
7. **`06bf136` / `292efbc` / `5811374` / `4fe683a` / `a2cd6d5`** ❌
   (trilogi "apakah parameter dasar sudah optimal", SEKARANG TUNTAS
   PENUH) — RSI/MFI/CMF/ATR/BB window (14/14/20/14/20): TIDAK ada yang
   lebih baik (25 kandidat, semua lebih buruk -- ini adalah default
   industri Wilder/`ta` library, bukan pilihan ad-hoc). ADX/RVOL/VWAP/
   structure window: sama, tidak ada yang menang. Regime classification
   threshold: 10 dari 11 kandidat lebih buruk, 1 nyaris sama (bukan
   temuan nyata, di dalam noise fold-to-fold). relative_strength_20d_pct
   / sector_relative_strength_20d_pct window (`4fe683a`, 2026-09-16):
   sama, tidak ada yang menang (window=40 tampak +0,07pp tapi lebih
   kecil dari ambang noise yang sudah ditetapkan di atas) -- ditemukan
   & diperbaiki bug nyata di tengah jalan (kolom sektor 100% NaN akibat
   index tanggal salah). pattern_similarity window/threshold
   (`a2cd6d5`, 2026-09-16) — kasus paling menarik: karena biayanya
   kuadratik (~3 jam/kandidat di skala penuh, bukan 541 detik seperti
   yang tertulis salah di docstring modulnya sendiri -- itu angka
   algoritma cKDTree yang DITOLAK, bukan yang dipakai), dipakai skema
   dua tahap (screening di subsample 350 ticker → konfirmasi cuma
   pemenang di skala penuh). Screening SEMPAT menunjukkan window=15
   unggul +2,56pp (jauh di atas noise level test lain) -- begitu
   dikonfirmasi di skala 900 ticker penuh, ternyata JUSTRU -0,81pp DI
   BAWAH baseline. Bank kecil membuat window pendek terlihat menang
   secara palsu (lebih banyak match "kebetulan" saat match asli langka).
   Skema dua tahap berhasil menangkap false positive ini sebelum jadi
   keputusan produksi.
8. **`bcb7c8a` / `f08385f`** ❌ — Slope window (3d/5d/10d) dan pilihan
   MA (SMA50/EMA9-20): individual kelihatan menang, tapi begitu
   dikombinasikan JUSTRU LEBIH BURUK dari baseline (klasik jebakan
   "one-variable-at-a-time" -- noise, bukan sinyal nyata). Tidak
   diadopsi.
9. **`bf9d133`** ❌ — Random search 40 konfigurasi hyperparameter
   (sebelum trilogi di atas) -- pemenang versi fold-average TIDAK
   terbukti begitu diverifikasi pooled. Menemukan bug metodologi: rata-
   rata fold TANPA bobot bisa menyesatkan (fold kecil disamakan
   bobotnya dengan fold besar) -- sejak itu SEMUA riset di atas pakai
   pooled Wilson LB, bukan rata-rata fold.
10. **`cca093f`** 📊 (hari ini) — Profit factor + frekuensi sinyal +
    horizon dibandingkan di kelima config. Win-rate saja menyesatkan;
    expectancy-per-hari (kecepatan modal) justru membalik urutan --
    config default (win-rate TERENDAH dari 5) menang di kecepatan
    modal, mengonfirmasi pilihan `top5_lift` sudah benar untuk investor
    yang optimasi kecepatan profit, bukan cuma hit-rate.

**Korelasi ke Thread B**: rally-speed feature (`ret_10d_pct`/
`ret_10d_atr_norm`, Thread B #1) yang lahir SEBELUM redesign target ini
ternyata tetap relevan -- muncul lagi sebagai kandidat kuat di riset
Turnaround v2 (Thread C, `153a0bb`).

---

## Thread B — Riset fitur untuk model Swing (feature engineering)

Format tiap butir: **fitur diuji → hasil**.

1. **`b0e21af`** ✅ **ret_10d_pct / ret_10d_atr_norm** (kecepatan rally
   10-hari, dinormalisasi ATR) -- dipicu pertanyaan Anda soal "saham
   naik cepat lalu dibanting". Diagnostik: win rate FLAT di semua
   tingkat kecepatan rally dalam zona BUY. Tapi sebagai fitur training:
   precision zona BUY naik 80,0%→81,1% (Wilson LB 77,1%→78,4%).
   **Diadopsi, retrain, jadi 44 fitur.**
2. **`1076470`** ❌ **Trend-acceleration ternormalisasi** (versi %-harga
   dari `ema20_slope_5d`/`ema20_accel_5d`/`sma50_slope_10d`/
   `macd_hist_slope_3d`/`macd_hist_accel_3d`, yang sebelumnya dikecualikan
   karena skala absolut Rupiah). Hasil: flat (+0,2pp, jauh di bawah
   rally-speed's +1,2pp) -- kemungkinan redundan dengan `rsi_slope`/
   `cmf_slope` yang sudah ada.
3. **`eac050e` → `669ed8c`** ❌ (sebagai fitur), ✅ (sebagai display) —
   **Wyckoff phase** (Accumulation/Markup/Distribution/Markdown).
   Ditolak sebagai fitur training (precision turun 71,9%→70,4%, LB
   -1,29pp lalu -1,67pp setelah redefinisi 4-fase). **Dipertahankan
   sebagai badge display-only** di Detail Saham, dengan caption
   eksplisit bahwa ini TIDAK membantu model.
4. **`3bc7d39`** ❌ **Fibonacci retracement** (fib_position, jarak ke
   61,8%, dekat-zona-fib). Hasil: -0,35pp, tidak diadopsi.
5. **`185797c` → `f5bed4a`** ❌ (sebagai fitur/rule), ✅ (sebagai
   display) — **Pivot multi-touch support/resistance**. Ditolak sebagai
   kriteria Momentum Screener (semua varian lebih buruk dari rule yang
   sudah ada). **Dipertahankan sebagai garis S/R di chart Detail
   Saham.**
6. **`2b92efc`** ❌ — Apakah fitur top-contributor Swing/Turnaround
   (mfi_14, overnight_gap_pct, ret_10d_atr_norm/pct, price_vs_sma50_pct,
   price_vs_vwap20_pct) berguna sebagai FILTER Momentum Screener? Tidak
   satu pun layak diadopsi -- konfirmasi kedua kalinya bahwa kontribusi
   XGBoost-gain (berguna dikombinasikan lewat ratusan split pohon) tidak
   berarti otomatis jadi threshold tunggal yang berguna.
7. **`4dfbf8b`** 📊 — Ablasi 5 kriteria Momentum Screener yang sudah
   live: tidak ada yang "mubazir" (regime=bottoming paling krusial).
   Varian coverage tinggi ditemukan (drop RVOL≥0,8 → 3,5x lebih banyak
   sinyal, LB masih 35,7% > null 30,6%) tapi tidak diimplementasi,
   dokumentasi saja.
8. **`c4e70be` / `b04987b`** ❌ — Ide StochRSI "early-reversal" (MACD
   histogram merah + EMA9<MA20 ≥9 hari + StochRSI %K silang %D + volume
   >MA20). Tidak ada edge sama sekali di kedua definisi target
   (triple-barrier maupun "sentuh +5%"). StochRSI-nya sendiri malah di
   BAWAH acak.

**Prinsip yang berulang muncul di Thread A & B**: kalau kemenangan
individual sebuah kandidat "hilang" begitu dikombinasikan dengan kandidat
lain yang juga menang sendiri-sendiri, itu tanda overfitting/noise, bukan
sinyal nyata (lihat `bcb7c8a`, `f08385f`, `bf9d133`). Metodologi combined-
recheck ini sekarang otomatis di setiap skrip riset baru.

---

## Thread C — Turnaround: dari perbaikan ke pensiun

1. **`bdb0a98`** 📊 — User minta Turnaround dipersingkat dari 6 bulan
   jadi maks 3 bulan, TIDAK dari regime bottoming/early_reversal. Hasil
   mengejutkan: regime terbaik untuk target ini adalah **overextended**
   (momentum continuation), bukan mean-reversion -- cerita yang
   sepenuhnya berbeda dari filosofi Momentum Screener (10 hari).
2. **`153a0bb` → `67b2e29` → `460f865`** 📊 — 3 ronde pencarian
   kondisi tambahan (rally-speed features, lalu VWAP/money-flow atas
   permintaan eksplisit user). Kombo terbaik: `price_vs_vwap20_pct>5`
   (LB 37,9%, n=5050) di atas regime overextended/bullish. Magnitude
   target dituning: **+20%/-10%/60 hari** direkomendasikan.
3. **`e50dc5e`** 📊 — Head-to-head: Swing (64,4% LB 57,7%) vs Turnaround
   v2 usulan (49,0% LB 42,0%) vs Turnaround lama (32,8% LB 26,6%), semua
   diukur ke target Turnaround v2. Turnaround v2 jauh lebih baik dari
   yang lama, TAPI Swing (model terlatih, bisa memilih dari ~900 saham)
   tetap menang telak walau diukur di target yang bukan target aslinya.
4. **`6b1b789`** ❌ — Ide berbeda: "bottoming/sideways bangun pagi →
   bagger 3 bulan". Ditolak keras -- median return aktual hari ke-60
   justru NEGATIF (-2,0%) walau "hit rate touch +20%" kelihatan 45,8%
   (itu cuma spike sementara yang hilang lagi, bukan re-rating nyata).
5. **`eafceba`** 🗑️ — **Keputusan final user**: hentikan pengembangan
   Turnaround (baik lama maupun v2), fokus ke Swing. Halaman dihapus
   dari app, tapi SEMUA kode/model/skrip riset dipertahankan di git
   (tidak hilang, hanya tidak aktif).

---

## Thread D — Manajemen risiko (kasus nyata memicu riset)

1. **`4123084`** 📊 — Dipicu kasus nyata LUCY (BUY di RSI 76,6/regime
   overextended, jatuh -33,7% dalam 5 hari). Ternyata **93% dari semua
   sinyal BUY** memang regime overextended -- bukan kasus langka.
   Tier non-overextended (7%) menang jauh lebih sering (LB 91,4% vs
   77,0%) tapi filter keras akan memangkas volume sinyal terlalu besar.
2. **`2725d16`** ✅ — Solusi non-destruktif: badge "Keyakinan Tinggi"
   untuk 7% tier yang lebih baik (bukan filter/blokir), plus catatan
   risiko suspensi umum.
3. **`59dde6f` → `c5c6f48`** — Riset risiko suspensi saham. v1 salah
   arah total (cari GAP tanggal, padahal saham suspend di IDX tetap
   punya baris harga tapi BEKU/volume=0) -- v2 dikoreksi, ketemu rate
   nyata **~2,25% dari sinyal BUY** (1 dari ~44) diikuti suspensi,
   dikonfirmasi lewat kasus nyata SAFE.
4. **`574f2e9`** ✅ — Kasus nyata SAFE (sudah suspend tapi masih tampil
   di Top 25 Swing). Semua screener sekarang exclude saham yang
   terdeteksi suspend (2 hari beruntun OHLC beku + volume 0).

**Catatan blind spot yang masih terbuka** (belum diperbaiki, baru
didokumentasikan di `59dde6f`/`c5c6f48`): label triple-barrier di
SELURUH backtest project ini diam-diam MENGECUALIKAN kasus "tidak
menyentuh target maupun stop" -- ini persis apa yang terjadi kalau harga
beku karena suspend. Jadi sejumlah kecil sinyal BUY historis yang
sebenarnya "gagal karena suspend" mungkin tidak terhitung sebagai
kerugian di angka precision manapun di seluruh project. Belum diaudit
ulang.

---

## Thread E — Fitur baru: Posisi Saya + UX

1. **`399f980`** ✅ — Fitur baru: tandai saham dibeli, tracking harian +
   early-warning SEBELUM stop-loss penuh kena. Rule early-warning
   dibacktest dulu (bukan diasumsikan): ANY_2 dari {underwater, close<
   EMA9, RSI slope<0, CMF<0} menangkap 5,1% kerugian historis lebih
   awal. Caveat jujur: 68,7% kerugian kena stop di HARI PERTAMA --
   tidak ada hari sebelumnya untuk memperingatkan mayoritas kasus.
2. **`d121aa4`** ✅ — Badge status per posisi (target_hit/stop_hit/
   expired/warning/under_pressure/on_track), diurutkan by urgensi.
3. **`f4064a7`** ✅ — Tombol "Tandai Beli" diperluas ke halaman Swing
   (sebelumnya cuma Detail Saham, dan cuma kalau sinyalnya BUY).
   Bug nyata ditemukan+diperbaiki: SQLite menolak `pd.Timestamp` sebagai
   nilai kolom Date.
4. **`c4a4d9e`** ✅ — Snapshot kondisi Swing (probability/regime/
   Wyckoff/target) DIREKAM saat klik "Tandai Beli", bukan dilihat ulang
   nanti -- supaya riwayat tetap benar walau model diretrain di
   kemudian hari.
5. **`96fe6b6`** ✅ — Fix bug: tanggal yang tercatat adalah tanggal
   PREDIKSI (bisa 3 hari lalu), bukan tanggal user klik hari ini.

Pendukung lain minggu ini: filter likuiditas opsional (`3b237b1`, TIDAK
default aktif -- terbukti secara statistik memangkas win-rate Swing
kalau dipaksa aktif), chart harga IHSG di Home dengan beberapa fix bug
(`5d6b2b7`/`1a9c114`/`2732de0`), scheduler jadi lebih robust
(`2b155e1`/`5b106df`/`2a8ecc4`), dan info model kini pakai precision
pooled bukan rata-rata fold (`b8f0ae8`).

---

## Pertanyaan cepat "apakah kita sudah coba X?"

| Ide | Sudah dicoba? | Hasil |
|---|---|---|
| Fibonacci retracement | ✅ ya | ❌ ditolak (`3bc7d39`) |
| Wyckoff phase sebagai fitur model | ✅ ya | ❌ ditolak, jadi display-only (`eac050e`) |
| Pivot support/resistance sebagai kriteria | ✅ ya | ❌ ditolak, jadi display-only (`185797c`) |
| StochRSI early-reversal | ✅ ya | ❌ tidak ada edge (`c4e70be`, `b04987b`) |
| Filter likuiditas keras | ✅ ya | ❌ menurunkan win-rate Swing kalau dipaksa (`3b237b1`) |
| Ganti algoritma dari XGBoost | ✅ ya (5 algoritma) | ❌ XGBoost tetap terbaik, tipis dari LightGBM (`b29d925`) |
| Hyperparameter Swing default | ✅ ya (40+ config) | ✅ ketemu perbaikan nyata, sudah live (`2535081`) |
| Window RSI/MFI/CMF/ATR/BB | ✅ ya | ❌ sudah optimal (default industri) (`06bf136`) |
| Bahaya suspend/freeze saham | ✅ ya | Rate nyata ~2,25% ditemukan, dimitigasi via exclusion (`c5c6f48`, `574f2e9`) |
| Turnaround v2 (3 bulan, +20%) | ✅ ya | Lebih baik dari Turnaround lama, TAPI kalah dari Swing -- Turnaround dipensiunkan (`eafceba`) |
| Window relative_strength/sector_relative_strength (vs 20 hari) | ✅ ya | ❌ sudah optimal, tidak ada yang menang (`4fe683a`, 2026-09-16) |
| Window/threshold pattern_similarity (vs 20/10/0,85) | ✅ ya (2 tahap: screening + konfirmasi) | ❌ sudah optimal -- screening SEMPAT menunjukkan window=15 menang, tapi TERBANTAH di konfirmasi skala penuh (artefak subsample) (`a2cd6d5`, 2026-09-16) |

---

*Dokumen ini mencakup commit `e75cd48..a2cd6d5` (2026-09-08 s/d
2026-09-16). Untuk pekerjaan setelah tanggal ini, jalankan
`git log --oneline a2cd6d5..HEAD` dan pertimbangkan menambah thread baru
di sini alih-alih membuat dokumen terpisah lagi.*
