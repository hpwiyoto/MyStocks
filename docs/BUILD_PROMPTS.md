# MyStocks — Prompt Pembangunan Bertahap

Dokumen ini berisi prompt siap-pakai untuk membangun MyStocks part-by-part, sesuai arsitektur dan keputusan yang sudah dibahas (Colab = research, Codespaces = build, VPS = production; yfinance sebagai satu-satunya sumber data; XGBoost vs LightGBM via walk-forward; target probabilistik; UI Streamlit profesional).

**Riwayat keputusan:** Stooq (fallback OHLCV) dan data resmi IDX (foreign flow via impor CSV manual) sempat dicoba di Fase 1, tapi di-drop — keduanya menambah kompleksitas operasional (proteksi anti-bot di sisi Stooq/IDX, dan kebutuhan impor manual rutin) yang tidak sepadan untuk tahap ini. Scope disederhanakan jadi yfinance-only. Fitur money-flow (CMF/OBV/MFI) tetap ada di Fase 2, dihitung murni dari OHLCV.

**Prinsip pemakaian — baca dulu sebelum mulai:**
1. Jalankan **satu fase per sesi**. Jangan minta gabungan beberapa fase sekaligus — ini sengaja dipisah supaya AI (dan Anda) tidak kehilangan konteks atau membuat asumsi yang bias/prematur.
2. **Review hasil setiap fase** sebelum lanjut ke fase berikutnya. Tiap prompt di bawah diakhiri instruksi eksplisit "tunjukkan hasil untuk direview" — jangan lewati ini.
3. Kalau memulai di sesi/lingkungan baru (misalnya chat Colab yang terpisah dari Claude Code ini), tempelkan **Context Anchor** di bawah ini dulu supaya konteks tidak hilang.
4. Setiap prompt fase sudah membatasi scope secara eksplisit ("JANGAN lakukan X di fase ini") — ini untuk mencegah AI overbuild atau lompat ke fase lain secara diam-diam.

---

## Context Anchor (tempel di awal sesi baru bila perlu)

```
Saya sedang membangun "MyStocks", aplikasi Stock AI/screener untuk saham Indonesia (IDX).
Arsitektur: Google Colab untuk riset (EDA, feature testing, ML experiment, backtest) — Codespaces
untuk membangun aplikasi produksi — VPS + Managed MySQL untuk production run. Colab TIDAK
pernah dipakai sebagai backend/database permanen.

Sumber data: yfinance sebagai satu-satunya sumber OHLCV+fundamental (ticker format KODE.JK),
dengan retry/backoff otomatis. Stooq dan data resmi IDX (foreign flow) sempat dicoba lalu
sengaja di-drop karena menambah kompleksitas operasional (anti-bot protection, kebutuhan impor
manual rutin) yang tidak sepadan untuk tahap ini — fokus disederhanakan ke yfinance saja.

Model kandidat: XGBoost dan LightGBM, dipilih lewat walk-forward backtest, bukan asumsi awal.
Target prediksi didefinisikan probabilistik, contoh: P(harga naik ≥5% dalam 10 hari trading
sebelum stop-loss -2.5%) — bukan prediksi harga persis.

Feature engineering mencakup: slope/akselerasi indikator (bukan cuma nilai mentah), market
regime (bearish/bottoming/sideways/accumulation/early reversal/bullish/overextended) sebagai
fitur kategorikal eksplisit, dan historical pattern similarity sebagai fitur tambahan.

Kita membangun ini bertahap, fase demi fase. Fase yang sedang saya minta adalah fase berikut:
[tempel prompt fase yang relevan di bawah sini]
```

---

## FASE 0 — Fondasi Proyek ✅ (sudah selesai)

**Tujuan:** kerangka repo saja, tanpa logic apapun.

```
Bangun fondasi proyek MyStocks dengan struktur folder berikut:
- /data          (raw & processed data, di-gitignore)
- /pipeline      (data ingestion: yfinance)
- /features      (feature engineering modules)
- /models        (training script & model artifacts)
- /engine        (prediction & decision engine)
- /app           (Streamlit application)
- /notebooks     (referensi/ekspor dari Colab)
- /scripts       (utility & cron scripts)
- /docs

Buat requirements.txt awal (pandas, numpy, yfinance, sqlalchemy, pymysql, streamlit, plotly,
python-dotenv), .env.example, .gitignore yang sesuai (data mentah, .env, __pycache__, model
artifact besar), dan perbarui README.md dengan ringkasan arsitektur 3-lingkungan
(Colab/Codespaces/Production).

JANGAN implementasikan logic apapun di fase ini — murni skeleton proyek. Setelah selesai,
tunjukkan struktur folder final untuk saya review sebelum saya minta fase berikutnya
(Data Pipeline).
```

---

## FASE 1 — Data Pipeline ✅ (sudah selesai, disederhanakan jadi yfinance-only)

**Tujuan:** data mentah masuk ke MySQL. Tanpa feature engineering.

**Riwayat keputusan** (ditemukan lewat pengujian langsung, bukan asumsi): rencana awal mencoba
Stooq sebagai fallback dan data resmi IDX (foreign buy/sell/frequency) via impor CSV manual.
Stooq ternyata dilindungi proof-of-work anti-bot dan endpoint resmi IDX dilindungi Cloudflare
managed challenge — keduanya sudah dicoba dengan browser automation asli (Playwright, tanpa
stealth/evasion) dan tetap terblokir, dan menembusnya butuh stealth-evasion yang sengaja tidak
dibangun. Setelah dicoba, impor manual CSV foreign_flow dinilai menambah kompleksitas
operasional yang tidak sepadan, sehingga scope disederhanakan: **yfinance saja**, tanpa fallback
provider dan tanpa foreign_flow. Fitur money-flow tetap tersedia di Fase 2 lewat CMF/OBV/MFI
yang dihitung murni dari OHLCV.

```
Bangun modul data pipeline di /pipeline:

yfinance untuk OHLCV + fundamental dasar saham IDX (ticker format KODE.JK, contoh BBCA.JK),
dengan retry/backoff otomatis saat gagal/timeout dan logging yang jelas.

Simpan ke skema MySQL dengan tabel terpisah:
- stocks         (master data ticker)
- price_history  (OHLCV harian, kolom source_provider)

Sertakan mekanisme update incremental (bukan re-download seluruh histori tiap run) dan logging
yang jelas. JANGAN membuat feature engineering atau model di fase ini.

Tunjukkan skema tabel final dan contoh hasil ingest untuk satu ticker (misalnya BBCA) untuk
saya verifikasi sebelum lanjut ke fase Feature Engineering.
```

---

## FASE 2 — Feature Engineering Engine ✅ (sudah selesai)

**Tujuan:** ubah data mentah jadi feature table. Tanpa training model.

**Catatan scope:** dieksplorasi langsung ke yfinance (2026-08-24) dan ternyata `.info` untuk
saham IDX terisi kaya (157 field untuk BBCA.JK: valuasi, profitabilitas, dividend, kepemilikan,
target analis), `.quarterly_financials`/`.balance_sheet` juga tersedia, dan index IHSG (`^JKSE`)
bisa di-fetch via yfinance juga. Semua ini gratis dari sumber yang sama (yfinance), tidak
menambah sumber data baru — jadi scope Fase 2 diperluas untuk memanfaatkannya di samping fitur
teknikal murni.

**Catatan implementasi** (ditemukan lewat pengujian, bukan asumsi):
- Base indikator (RSI/MACD/Bollinger/ATR/CMF/MFI/OBV) pakai library `ta`, BUKAN `pandas-ta` —
  pandas-ta diam-diam menurunkan numpy yang sudah dipin (lewat dependency `numba`) dan
  menghasilkan RSI pertama bernilai 0.0 yang mencurigakan saat diuji dengan data acak.
- `ta`'s `AverageTrueRange` crash (IndexError) kalau diberi <14 baris data — ditambahkan guard
  `MIN_ROWS_FOR_TECHNICAL_FEATURES=60` di orkestrator, ticker dengan histori pendek (baru
  listing) di-skip dengan warning, bukan error.
- Pattern similarity dihitung within-ticker (bukan cross-ticker) via correlation matrix
  ter-vektorisasi, dengan 2 constraint yang diverifikasi manual pada data nyata: TIDAK ada
  overlap antar window yang dibandingkan, dan TIDAK ada lookahead (outcome historis suatu pola
  hanya dipakai kalau sudah "diketahui" pada tanggal fitur itu dihitung).
- `dividendYield` dari yfinance ternyata berskala 0-100 (persen), BUKAN 0-1 seperti
  `trailingAnnualDividendYield` — dikonfirmasi silang manual, didokumentasikan di kode supaya
  tidak salah dikalikan 100 lagi di fase berikutnya.
- Fundamental snapshot disimpan di tabel terpisah (`feature_fundamental_snapshot`, satu baris
  per stock per tanggal snapshot) karena datanya point-in-time, bukan time series harian seperti
  `feature_daily`.

```
Bangun feature engineering engine di /features dari data di MySQL (fase sebelumnya). Cakupan:

Teknikal (dari price_history / OHLCV):
- Trend: SMA/EMA multi-period + slope + acceleration + distance_from_price
- Momentum: RSI, MACD (line/signal/histogram) + slope/acceleration/3d-5d change
- Volume: RVOL + volume slope/acceleration
- Money flow: CMF, OBV, MFI + slope
- Volatility: ATR%, Bollinger Band width + perubahannya
- Market structure: higher-high/lower-low, distance to support/resistance
- Market regime: klasifikasi kategorikal (bearish/bottoming/sideways/accumulation/
  early reversal/bullish/overextended) — jelaskan aturan/pendekatan yang dipakai
- Historical pattern similarity: similarity_score, similar_pattern_count,
  historical_win_rate berbasis pola N-hari

Fundamental & konteks pasar (dari yfinance .info / .quarterly_financials / index ^JKSE — data
snapshot, bukan time series harian, jadi desain penyimpanannya boleh beda dari feature teknikal):
- Valuasi: trailingPE, forwardPE, priceToBook, pegRatio
- Profitabilitas: returnOnEquity, returnOnAssets, profitMargins, operatingMargins
- Dividend: dividendYield, payoutRatio
- Ukuran & kepemilikan: marketCap, heldPercentInsiders, heldPercentInstitutions
- Sentimen analis: recommendationMean, upside % (targetMeanPrice vs harga sekarang)
- Relative strength vs IHSG (^JKSE): return saham dikurangi return index pada window yang sama

Simpan hasil sebagai tabel feature_daily di MySQL dengan kolom feature_version supaya
perubahan feature set di masa depan tidak menimpa histori lama. JANGAN training model
di fase ini.

Tunjukkan daftar lengkap feature yang dihasilkan + contoh output untuk satu ticker
supaya saya bisa cek sebelum lanjut ke fase ML Research.
```

---

## FASE 3 — ML Research (dikerjakan di Google Colab, bukan Codespaces) ✅ (sudah selesai)

**Tujuan:** validasi model & feature secara ilmiah sebelum masuk production code.

**Catatan implementasi** (ditemukan lewat pengujian nyata di Codespaces sebelum ditranskripsi ke
notebook — bukan ditulis lalu berharap jalan di Colab):
- Colab tidak bisa menjangkau MySQL lokal/dev, jadi dibuat `scripts/export_for_colab.py` yang
  mengekspor `feature_daily` + `price_history` (high/low/close) ke Parquet untuk diupload manual.
- Fungsi triple-barrier labeling (`P(+5% sebelum -2.5% dalam 10 hari)`) diuji dengan 8 test case
  tangan (termasuk boundary: tepat hari ke-10, kedua barrier di hari sama, data future tidak
  cukup) — konvensi: stop-loss menang kalau ambigu, dan hasil "tidak resolve dalam horizon"
  di-exclude (NaN) bukan dipaksa jadi label 0/1.
- Walk-forward split pakai EMBARGO — tanpa ini, baris training dekat batas fold akan memakai
  label yang resolusinya "mengintip" ke periode test (karena label butuh 10 hari ke depan).
  Diverifikasi matematis: batas embargo persis pas, dikonfirmasi dengan membandingkan label dari
  data penuh vs data terpotong dan hasilnya identik di semua fold.
- `historical_win_rate` NaN di ~44% baris (hari tanpa pola historis mirip) — diisi netral 0.5 +
  flag `has_similar_pattern`, bukan drop baris (drop akan membuang hampir separuh dataset).
- Bug ditemukan & diperbaiki: baris `feature_daily` dari SQL tidak terurut kronologis lintas
  ticker (ter-grup per ticker) — `max_drawdown` yang path-dependent jadi salah kalau tidak
  di-sort dulu. `total_return`/`win_rate`/`sharpe` tidak terpengaruh (order-invariant secara
  matematis), tapi drawdown pasti salah tanpa fix ini.
- Model tidak diberi `random_state` awalnya → hasil sedikit beda tiap run (non-deterministic).
  Diperbaiki, diverifikasi dengan diff 2 run berturut-turut → identik byte-per-byte.
- Notebook diverifikasi dengan cara diekstrak dan dieksekusi persis seperti Colab akan
  menjalankannya (bukan cuma dibaca sekilas) — hasilnya identik dengan skrip standalone yang
  sudah diuji terpisah.
- **Hasil run pertama** (10 ticker, 5 tahun): ROC-AUC ketiga model ~0.51-0.53 (sedikit di atas
  acak — sinyal sehat untuk dataset sekecil ini, bukan tanda kegagalan). Temuan penting: SHAP
  menunjukkan fitur skala-absolut (ema_9, sma_200, obv) mendominasi, bukan fitur ternormalisasi —
  indikasi model mungkin sebagian "menghafal saham" lewat level harga.
- **Iterasi 2** (kolom skala-absolut dikeluarkan dari feature set — sma_20/50/200, ema_9/20/50,
  macd/macd_signal/macd_hist+turunannya, volume_slope_5d, obv+turunannya): hipotesis terbukti
  benar. SHAP sekarang didominasi fitur ternormalisasi (atr_pct_14, bb_width_pct,
  price_vs_sma50_pct, dst). ROC-AUC tetap ~0.51-0.53 (wajar), tapi metrik trading membaik jelas:
  Sharpe XGBoost -0.16 → **+0.86**, profit factor 0.98 → 1.45, max drawdown -50% → -28%.
  XGBoost tampak paling menarik di iterasi ini, tapi tetap TIDAK diputuskan sebagai model final
  di sini — itu keputusan Anda sebelum masuk Fase 4.
- **Keputusan final (setelah review)**: XGBoost. Dilatih ulang pada seluruh data historis (bukan
  cuma satu fold) via cell "Export model produksi" di notebook, disimpan sebagai
  `models/direction_xgboost_v1.json` + `_metadata.json` (feature_cols, base_rate, target_pct,
  stop_pct — dipakai langsung oleh Fase 4, bukan diduplikasi/di-hardcode ulang).

```
Fase ini dikerjakan di Google Colab sesuai pembagian lingkungan riset vs build. Bantu saya
menyiapkan notebook/script untuk:

1. Load feature_daily dari MySQL (koneksi read-only).
2. Definisikan target probabilistik: P(harga naik ≥5% dalam 10 hari trading sebelum stop-loss
   -2.5%) sebagai label biner dari data historis.
3. Eksperimen paralel: baseline (mis. logistic regression) vs XGBoost vs LightGBM.
4. Walk-forward validation yang time-aware (bukan random train-test split) untuk menghindari
   lookahead bias.
5. Evaluasi dengan metrik ML (precision/recall/F1/ROC-AUC/calibration) DAN metrik trading
   (return, win rate, max drawdown, Sharpe ratio, profit factor).
6. Feature importance/SHAP untuk feature selection — jangan pakai semua feature mentah-mentah,
   evaluasi kontribusi masing-masing.

JANGAN memutuskan model final di fase ini — cukup laporkan hasil komparasi dan rekomendasi.
Saya akan review sebelum model pemenang di-commit ke Codespaces di fase berikutnya.
```

---

## FASE 4 — Prediction & Decision Engine ✅ (sudah selesai)

**Tujuan:** model terpilih jadi output actionable. Tanpa UI.

**Catatan implementasi** (ditemukan lewat pengujian, bukan asumsi):
- `xgboost` dipin ke `~=2.0.0` (BUKAN `~=2.0`, yang ternyata tetap resolve ke 2.1.4 — pelajaran
  soal operator `~=`) — versi 2.1+ membundel `nvidia-nccl-cu12` (~300MB) secara default meski
  cuma dipakai untuk inference CPU, dan xgboost 2.0.3 tidak membawa itu sama sekali.
- Bug nyata: `XGBClassifier.save_model()` (wrapper sklearn) crash
  (`TypeError: _estimator_type undefined`) pada kombinasi xgboost 2.0.3 + scikit-learn 1.9 yang
  lebih baru. Solusi: `model.get_booster().save_model()` untuk simpan,
  `xgb.Booster().load_model()` + `predict(DMatrix)` untuk load — diverifikasi hasil prediksi
  identik dengan wrapper sklearn.
- `.gitignore` awalnya mengecualikan `models/*.json` (dari Fase 0, sebelum ada artifact
  sungguhan) — ditambahkan pengecualian eksplisit untuk `direction_xgboost_v1.json` +
  `_metadata.json` supaya benar-benar ter-commit, bukan cuma pola umum yang di-drop.
- `engine/model.py` merekonstruksi ulang preprocessing training (imputasi historical_win_rate,
  flag has_similar_pattern, one-hot regime) untuk SATU baris live — bukan pakai
  `pd.get_dummies` langsung (yang cuma akan menghasilkan kolom untuk regime baris itu saja,
  bukan seluruh kolom regime_* yang diharapkan model).
- Diuji: ticker tanpa `feature_daily`, ticker dengan fitur yang hilang (NULL) — keduanya
  di-skip dengan warning jelas berisi daftar kolom yang hilang, bukan crash atau prediksi dari
  data rusak. Idempotency diverifikasi (re-run tidak membuat duplikat).
- **Recheck integrasi (2026-08-25)**: dites di clone bersih dari nol (venv baru, MySQL baru,
  `ingest_price` → `build_features` → `predict` end-to-end) — semua konsisten, `feature_cols` di
  metadata model cocok 100% dengan skema `feature_daily` aktual, semua nilai `regime` di data
  tercakup kolom `regime_*` model (tidak ada yang silently ter-encode nol). Satu gap ditemukan:
  kalau `feature_daily`/`price_history` belum ada sama sekali (DB kosong, belum jalankan Fase
  1/2), `engine.predict` tetap tidak crash total tapi mencetak 10× traceback SQL penuh yang
  membingungkan. Diperbaiki dengan pengecekan tabel eksplisit di awal `run()` — sekarang
  menghasilkan satu pesan jelas ("jalankan pipeline.ingest_price dan features.build_features
  dulu") sebelum masuk loop ticker.

```
Model terpilih dari hasil Colab sudah saya berikan (artifact/parameter). Bangun /engine di
Codespaces:

1. Prediction engine yang me-load model terpilih dan menghasilkan probability per ticker
   per hari.
2. Mulai dengan satu model Direction dulu (bukan langsung 4 model spesialis
   Direction/Return/Risk/Time) supaya tidak overbuild — model tambahan menyusul di fase
   terpisah nanti bila diperlukan.
3. Decision engine yang mengonversi probability menjadi output actionable: BUY/WATCH/AVOID +
   Entry/SL/TP/R:R, dengan aturan risk management yang dijelaskan asumsinya (jangan hardcode
   angka tanpa justifikasi).
4. Simpan hasil prediksi harian ke tabel predictions di MySQL.

JANGAN membangun UI di fase ini. Tunjukkan contoh output prediksi untuk beberapa ticker
supaya saya validasi logikanya sebelum masuk ke fase UI.
```

---

## FASE 5 — Streamlit Application (UI) ✅ (sudah selesai)

**Tujuan:** presentation layer yang bersih, smooth, dan terasa profesional.

**Keputusan (2026-08-25)**: user sempat menanyakan apakah perlu tombol "training ulang" di
aplikasi. Diputuskan TIDAK — bertentangan dengan pemisahan Colab (riset/training, butuh review
manusia atas walk-forward/SHAP) vs Codespaces/app (serving). Sebagai gantinya, tambahkan
**indikator read-only** di UI: umur model (`trained_at` dari metadata) dan berapa banyak data
baru yang terkumpul sejak training (`n_training_rows` vs jumlah baris `feature_daily` sekarang)
— pengingat kapan sebaiknya retrain manual di Colab, bukan tombol aksi.

**Catatan implementasi** (diverifikasi dengan Playwright headless — screenshot & klik navigasi
sungguhan, bukan cuma baca kode):
- Menu native Streamlit (`app/pages/`) dipakai untuk sidebar nav, dark theme diatur di
  `.streamlit/config.toml`, badge/kartu custom di `app/style.py`.
- Bug data ditemukan lewat UI, bukan lewat cek kode: P/B ADRO tampil **15470.59** — dilacak
  sampai sumbernya (yfinance `bookValue=0.17` untuk harga 2630, kemungkinan glitch data Yahoo
  Finance), BUKAN bug di pipeline kita. Ditambahkan guard tampilan (`safe_ratio()` di
  `app/style.py`) yang menampilkan "N/A*" + penjelasan untuk rasio valuasi di luar rentang wajar,
  daripada menampilkan angka menyesatkan.
- `use_container_width=True` (dipakai di 4 tempat) ternyata sudah melewati tanggal deprecation
  Streamlit (2025-12-31) — diganti `width="stretch"` di semua pemanggilan.
- Glitch dev-only: menambah nama baru ke modul yang sudah ke-import (`app.style`) tidak
  otomatis ter-refresh oleh hot-reload Streamlit — perlu restart proses dev server, bukan
  bug kode.
- Filter (keputusan/regime/probabilitas minimum) diuji interaktif — jumlah kartu yang tampil
  berubah sesuai filter dan konsisten dengan data asli.

```
Bangun aplikasi Streamlit di /app yang menyajikan hasil dari prediction engine. Spesifikasi UI:

- Layout: halaman utama = dashboard screener (tabel/card ranking saham berdasarkan probability
  tertinggi), halaman detail per saham (candlestick chart interaktif via Plotly + overlay
  indikator + panel prediction/regime/pattern similarity).
- Visual: palet warna konsisten, dark mode sebagai default (umum di dashboard trading) dengan
  dukungan light mode, tipografi dengan hierarki jelas (angka penting seperti probability/return
  ditonjolkan), gunakan card/container dengan spacing rapi, hindari tampilan padat/berantakan.
- Interaksi: filter & sorting responsif tanpa reload penuh (session_state dikelola dengan baik),
  loading indicator saat fetch data, transisi antar halaman yang mulus.
- Komponen kunci: badge warna untuk regime (hijau=bullish, merah=bearish, kuning=sideways),
  gauge/progress bar untuk probability, tabel screener yang bisa difilter (sektor, regime,
  minimum probability).
- /app hanya boleh memanggil hasil dari /engine dan MySQL — tidak ada logic pipeline/training
  di file UI.

Setelah selesai, jalankan aplikasi secara lokal, tunjukkan hasilnya (screenshot/deskripsi),
dan saya akan review UX sebelum masuk ke fase Deployment.
```

---

## FASE 6 — Production Deployment

**Tujuan:** aplikasi berjalan otomatis di VPS. Tanpa mengubah logic model/UI.

**Keputusan (2026-08-25)**: user menanyakan notifikasi kalau yfinance gagal/tidak update.
Sengaja digabung ke fase ini (bukan dibangun terpisah lebih awal) karena baru relevan begitu
pipeline berjalan otomatis tanpa pengawasan (poin 2 & 4 di bawah) — kalau masih dijalankan
manual, kegagalan sudah langsung terlihat di terminal. Saat implementasi, bedakan dua mode
kegagalan: (a) error eksplisit (koneksi gagal, exception — sudah ada retry+isolasi per-ticker
dari Fase 1) vs (b) "diam-diam basi" (fetch sukses tapi tanggal terbaru di price_history tidak
maju beberapa hari) — keduanya butuh deteksi terpisah, bukan cuma cek exception.

```
Siapkan deployment ke VPS + Managed MySQL untuk production (Colab dan Codespaces tidak dipakai
sebagai server production). Cakupan:

1. Containerize aplikasi (Dockerfile untuk pipeline+engine, dan untuk Streamlit app — atau
   docker-compose bila lebih sesuai).
2. Scheduled job (cron/APScheduler) untuk update data harian (Fase 1) dan regenerate prediction
   (Fase 4) otomatis setelah market close.
3. Environment variable & secret management (kredensial MySQL/API tidak boleh hardcoded).
4. Monitoring/alerting dasar bila pipeline gagal (misalnya notifikasi saat ingest data gagal
   berturut-turut).

JANGAN mengubah logic model/feature/UI di fase ini — murni operasional deployment. Tunjukkan
langkah deployment lengkap + checklist verifikasi sebelum aplikasi dianggap production-ready.
```
