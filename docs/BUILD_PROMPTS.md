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

## FASE 2 — Feature Engineering Engine

**Tujuan:** ubah data mentah jadi feature table. Tanpa training model.

```
Bangun feature engineering engine di /features dari data di MySQL (fase sebelumnya). Cakupan:

- Trend: SMA/EMA multi-period + slope + acceleration + distance_from_price
- Momentum: RSI, MACD (line/signal/histogram) + slope/acceleration/3d-5d change
- Volume: RVOL + volume slope/acceleration
- Money flow: CMF, OBV, MFI + slope (dihitung murni dari OHLCV yfinance)
- Volatility: ATR%, Bollinger Band width + perubahannya
- Market structure: higher-high/lower-low, distance to support/resistance
- Market regime: klasifikasi kategorikal (bearish/bottoming/sideways/accumulation/
  early reversal/bullish/overextended) — jelaskan aturan/pendekatan yang dipakai
- Historical pattern similarity: similarity_score, similar_pattern_count,
  historical_win_rate berbasis pola N-hari

Simpan hasil sebagai tabel feature_daily di MySQL dengan kolom feature_version supaya
perubahan feature set di masa depan tidak menimpa histori lama. JANGAN training model
di fase ini.

Tunjukkan daftar lengkap feature yang dihasilkan + contoh output untuk satu ticker
supaya saya bisa cek sebelum lanjut ke fase ML Research.
```

---

## FASE 3 — ML Research (dikerjakan di Google Colab, bukan Codespaces)

**Tujuan:** validasi model & feature secara ilmiah sebelum masuk production code.

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

## FASE 4 — Prediction & Decision Engine

**Tujuan:** model terpilih jadi output actionable. Tanpa UI.

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

## FASE 5 — Streamlit Application (UI)

**Tujuan:** presentation layer yang bersih, smooth, dan terasa profesional.

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
