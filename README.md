# MyStocks

Stock AI / Screener untuk saham Indonesia (IDX), dengan feature engineering, model probabilistik (XGBoost/LightGBM), dan dashboard Streamlit.

## Arsitektur

- **Google Colab** — research lab: EDA, feature testing, ML experiment, backtesting, pattern matching, model training. Tidak pernah dipakai sebagai backend/database permanen.
- **GitHub Codespaces (repo ini)** — factory/development: data pipeline, feature engine, prediction engine, aplikasi Streamlit, testing.
- **VPS + Managed MySQL** — production: aplikasi berjalan, scheduled data update, prediksi harian, alert.

Alur: eksperimen/validasi di Colab → terbukti bagus → commit ke GitHub → tarik ke Codespaces → integrasi ke production code → deploy ke VPS.

## Sumber data

- **yfinance** — sumber utama OHLCV + fundamental dasar (ticker format `KODE.JK`, contoh `BBCA.JK`).
- **Stooq** — fallback OHLCV historis tanpa API key.
- **Data resmi IDX** — foreign buy/sell, frequency, value transaksi sebagai fitur money-flow tambahan yang tidak dimiliki Yahoo Finance.

## Struktur proyek

```
data/        raw & processed data (gitignored)
pipeline/    data ingestion (yfinance, Stooq, IDX scraper)
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
