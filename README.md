# MyStocks

Stock AI / Screener untuk saham Indonesia (IDX), dengan feature engineering, model probabilistik (XGBoost/LightGBM), dan dashboard Streamlit.

## Arsitektur

- **Google Colab** — research lab: EDA, feature testing, ML experiment, backtesting, pattern matching, model training. Tidak pernah dipakai sebagai backend/database permanen.
- **GitHub Codespaces (repo ini)** — factory/development: data pipeline, feature engine, prediction engine, aplikasi Streamlit, testing.
- **VPS + Managed MySQL** — production: aplikasi berjalan, scheduled data update, prediksi harian, alert.

Alur: eksperimen/validasi di Colab → terbukti bagus → commit ke GitHub → tarik ke Codespaces → integrasi ke production code → deploy ke VPS.

## Sumber data

- **yfinance** — satu-satunya sumber data (OHLCV + fundamental dasar, ticker format `KODE.JK`, contoh `BBCA.JK`), dengan retry/backoff otomatis. Fokus sengaja disederhanakan ke yfinance saja — data resmi IDX (foreign buy/sell/frequency) sempat dicoba tapi di-drop karena butuh impor manual per hari (endpoint resminya dilindungi Cloudflare, Stooq dilindungi proof-of-work anti-bot, keduanya terbukti tidak bisa diotomasi tanpa stealth-evasion yang sengaja tidak dibangun) — kompleksitas itu tidak sepadan untuk saat ini. Fitur money-flow (CMF/OBV/MFI) tetap bisa dihitung nanti murni dari OHLCV tanpa data foreign flow.

## Struktur proyek

```
data/        raw & processed data (gitignored)
pipeline/    data ingestion (yfinance)
features/    feature engineering
models/      training script & model artifacts
engine/      prediction & decision engine
app/         aplikasi Streamlit
notebooks/   referensi/ekspor dari Colab
scripts/     utility & cron scripts
docs/        dokumentasi, termasuk BUILD_PROMPTS.md
```

Lihat [docs/BUILD_PROMPTS.md](docs/BUILD_PROMPTS.md) untuk rencana pembangunan bertahap (Fase 0–6).

## Setup

Python 3.12 (dikunci lewat `.devcontainer/devcontainer.json` saat dibuka di Codespaces).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # isi kredensial MySQL
```

## Data pipeline

Update harga saham (incremental, aman dijalankan berulang):

```bash
python -m pipeline.ingest_price
```

Ticker yang dilacak diatur di `pipeline/tickers.py` (`SEED_TICKERS`).

## Feature engineering

Hitung feature teknikal (`feature_daily`) + fundamental & relative strength vs IHSG (`feature_fundamental_snapshot`) dari data yang sudah di-ingest:

```bash
python -m features.build_features
```

Butuh minimal 60 hari price_history per ticker (indikator seperti ATR/RSI butuh window minimum; ticker yang lebih baru dari itu otomatis di-skip dengan warning, bukan error). Base indikator (RSI/MACD/Bollinger/ATR/CMF/MFI/OBV) memakai library `ta` — bukan `pandas-ta`, yang terbukti diam-diam menurunkan versi numpy yang sudah kita pin dan sempat menghasilkan nilai RSI yang mencurigakan saat diuji.

## ML Research (Fase 3, di Google Colab)

Colab tidak bisa menjangkau MySQL lokal/dev, jadi datanya diekspor dulu ke Parquet:

```bash
python -m scripts.export_for_colab
```

Menghasilkan `data/export_for_colab_features.parquet` dan `..._prices.parquet` — upload keduanya ke [notebooks/fase3_ml_research.ipynb](notebooks/fase3_ml_research.ipynb) di Colab. Notebook membandingkan baseline/XGBoost/LightGBM lewat walk-forward validation (dengan embargo anti-lookahead) dan **tidak** memutuskan model final — itu keputusan manual sebelum Fase 4.

Model produksi terpilih (**XGBoost**, `models/direction_xgboost_v1.json` + `_metadata.json`) dilatih pada seluruh data historis lewat sel "Export model produksi" di notebook yang sama.

## Prediction & Decision Engine (Fase 4)

Skor tiap ticker yang di-track pakai model Direction terlatih, hasilkan keputusan BUY/WATCH/AVOID + Entry/SL/TP:

```bash
python -m engine.predict
```

Aturan keputusan (di `engine/decision.py`, bukan angka sembarang):
- Entry/SL/TP diturunkan langsung dari `target_pct`/`stop_pct` yang sama dengan definisi label saat training (5% / 2.5%) — R:R tetap 2.0.
- **BUY**: probability ≥ 0.5. **WATCH**: probability ≥ base rate historis model tapi < 0.5. **AVOID**: di bawah base rate (tidak ada edge).

Hasil tersimpan di tabel `predictions` (upsert per `stock_code`+`date`+`model_version`, aman dijalankan berulang). Ticker tanpa `feature_daily` atau dengan fitur yang hilang di-skip dengan warning jelas, bukan crash atau prediksi dari data rusak.

## Aplikasi Streamlit (Fase 5)

```bash
streamlit run app/Home.py
```

- **Screener** (halaman utama) — kartu ranking saham berdasarkan probabilitas, badge warna untuk decision (BUY/WATCH/AVOID) & regime, filter di sidebar (keputusan, regime, probabilitas minimum).
- **Detail Saham** — candlestick chart (Plotly, overlay SMA20/50/200 + volume), Entry/SL/TP, panel pattern similarity & fundamental.
- **Info Model** — indikator umur model & jumlah data baru sejak training terakhir (pengingat retrain manual, bukan tombol aksi — lihat catatan Fase 5 di `docs/BUILD_PROMPTS.md`).

Tema warna diatur di `.streamlit/config.toml` (dark mode) + `app/style.py` (badge/kartu). `/app` murni presentation layer — hanya query MySQL & baca `models/*_metadata.json`, tidak ada logic pipeline/training di dalamnya.
