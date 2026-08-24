# MyStocks

Stock AI / Screener untuk saham Indonesia (IDX), dengan feature engineering, model probabilistik (XGBoost/LightGBM), dan dashboard Streamlit.

## Arsitektur

- **Google Colab** — research lab: EDA, feature testing, ML experiment, backtesting, pattern matching, model training. Tidak pernah dipakai sebagai backend/database permanen.
- **GitHub Codespaces (repo ini)** — factory/development: data pipeline, feature engine, prediction engine, aplikasi Streamlit, testing.
- **VPS + Managed MySQL** — production: aplikasi berjalan, scheduled data update, prediksi harian, alert.

Alur: eksperimen/validasi di Colab → terbukti bagus → commit ke GitHub → tarik ke Codespaces → integrasi ke production code → deploy ke VPS.

## Sumber data

- **yfinance** — satu-satunya sumber OHLCV + fundamental dasar (ticker format `KODE.JK`, contoh `BBCA.JK`), dengan retry/backoff otomatis.
- **Data resmi IDX** — foreign buy/sell, frequency, value transaksi, diimpor **manual** dari CSV yang diunduh sendiri lewat browser (lihat bawah). Endpoint resmi IDX dan Stooq sama-sama terbukti dilindungi anti-bot (Cloudflare / proof-of-work) saat diuji langsung, sehingga scraping otomatis sengaja tidak dibangun.

## Struktur proyek

```
data/        raw & processed data (gitignored)
pipeline/    data ingestion (yfinance, import CSV foreign flow)
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

Impor data foreign flow dari CSV yang diunduh manual dari idx.co.id (Data Pasar → Ringkasan Perdagangan → Download):

```bash
python -m pipeline.import_foreign_flow path/to/file.csv
```

Ticker yang dilacak diatur di `pipeline/tickers.py` (`SEED_TICKERS`).

Karena `foreign_flow` hanya bisa diperbarui manual, `python -m pipeline.ingest_price` **selalu** mengecek kesegarannya di akhir dan mencetak WARNING (di console + `data/logs/pipeline.log`) kalau ada saham yang datanya belum pernah diimpor atau sudah lewat 3 hari. Cek manual kapan saja: `python -m pipeline.alerts`.
