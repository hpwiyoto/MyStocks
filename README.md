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
