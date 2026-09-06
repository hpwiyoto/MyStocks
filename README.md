# MyStocks

Stock AI / Screener untuk saham Indonesia (IDX): dua model machine learning (Swing & Turnaround), satu screener aturan teknikal tervalidasi lewat backtest (Momentum Screener), halaman gabungan lintas-alat (Rekomendasi Emitten), dan dashboard Streamlit.

## Arsitektur

- **Google Colab** — research lab: EDA, feature testing, ML experiment, backtesting, pattern matching, model training. Tidak pernah dipakai sebagai backend/database permanen.
- **GitHub Codespaces** — factory/development (Docker + MySQL, sama seperti production): data pipeline, feature engine, prediction engine, aplikasi Streamlit, testing.
- **PC lokal (Windows, native, tanpa Docker)** — jalur dev alternatif yang setara: SQLite otomatis (zero-config), `scripts/start.ps1` menjalankan semuanya dengan satu perintah `start`. Kode aplikasi sama persis di kedua jalur — `pipeline/db.py` yang memilih SQLite vs MySQL berdasarkan environment, bukan cabang kode terpisah.
- **VPS + Managed MySQL** — production: aplikasi berjalan, scheduled data update, prediksi harian, alert.

Alur: eksperimen/validasi di Colab (atau backtest langsung di dev environment mana pun, lihat `scripts/*.py` — banyak keputusan terakhir divalidasi begini, bukan di Colab) → terbukti bagus → commit ke GitHub → deploy ke VPS.

**Sinkronisasi GitHub**: `scripts/start.ps1` sekarang otomatis `git pull --ff-only` (aman, tidak pernah menimpa perubahan lokal yang belum di-commit) lalu `git push` commit lokal yang belum ter-backup, setiap kali `start` dijalankan — PC lokal dianggap sebagai salinan utama, GitHub sebagai backup, bukan sebaliknya. Commit tetap manual (`git add`/`commit`), cuma push-nya yang otomatis.

## Sumber data

- **yfinance** — satu-satunya sumber data harga (OHLCV + fundamental dasar, ticker format `KODE.JK`, contoh `BBCA.JK`), dengan retry/backoff otomatis. Fokus sengaja disederhanakan ke yfinance saja untuk harga — data resmi IDX (foreign buy/sell/frequency) sempat dicoba tapi di-drop karena butuh impor manual per hari (endpoint resminya dilindungi Cloudflare, Stooq dilindungi proof-of-work anti-bot). Fitur money-flow (CMF/OBV/MFI) dihitung murni dari OHLCV tanpa data foreign flow.
- **RapidAPI "Indonesia Stock Exchange (IDX)"** — dicoba sebagai fitur tambahan untuk model Swing (`net_foreign_flow`, dinormalisasi z-score), diuji empiris lewat walk-forward validation, **tidak meningkatkan performa model** (ROC-AUC nyaris tidak berubah) sehingga **tidak dipakai sebagai input model**. Infrastrukturnya (`pipeline/idx_rapidapi_source.py`) tetap dipakai untuk melayani panel **Foreign Flow** yang tampil di Detail Saham sebagai pelengkap analisis manual (bukan input model) — diambil on-demand saat halaman dibuka, dengan quota tracking persisten supaya tidak melebihi batas gratis RapidAPI.

## Struktur proyek

```
data/        raw & processed data (gitignored)
pipeline/    data ingestion (yfinance, RapidAPI foreign-flow on-demand)
features/    feature engineering (teknikal, fundamental, regime, momentum screener rules)
models/      training script & model artifacts (Swing + Turnaround)
engine/      prediction & decision engine (Swing + Turnaround)
app/         aplikasi Streamlit
notebooks/   referensi/ekspor dari Colab
scripts/     utility, cron scripts, DAN skrip backtest/riset (lihat "Riset & Backtest" di bawah)
docs/        dokumentasi, termasuk BUILD_PROMPTS.md
```

Lihat [docs/BUILD_PROMPTS.md](docs/BUILD_PROMPTS.md) untuk rencana & catatan pembangunan bertahap (Fase 0–7+).

## Setup

Python 3.12 (dikunci lewat `.devcontainer/devcontainer.json` di Codespaces).

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # opsional -- lihat catatan database di bawah
```

**Database**: dev (Codespaces maupun PC lokal) memakai **SQLite** secara default (`data/mystocks.db`, dibuat otomatis, tanpa instalasi/service terpisah) -- `.env` tidak perlu diisi sama sekali untuk ini. Production (VPS) tetap MySQL lewat `docker-compose.yml` -- `pipeline/db.py` otomatis memilih MySQL hanya jika `MYSQL_HOST` di-set di environment, jadi kode yang sama jalan di semua tempat tanpa disunting.

### Windows tanpa Docker: cukup ketik `start`

Kalau bekerja di Windows secara native (bukan Codespaces/WSL), `scripts/start.ps1` menyiapkan semuanya otomatis: sinkron dengan GitHub (pull lalu push, lihat "Sinkronisasi GitHub" di atas), membuat `.venv` kalau belum ada, `pip install`, lalu menjalankan scheduler + aplikasi Streamlit masing-masing di jendela PowerShell tersendiri (minimized, jadi log tetap terlihat lewat taskbar & masing-masing bisa ditutup sendiri), menunggu sampai aplikasi benar-benar merespon, lalu membuka browser otomatis.

Sekali setup (menambahkan fungsi `start` ke PowerShell `$PROFILE`, hanya aktif saat direktori kerja ada di dalam folder repo ini):
```powershell
# jalankan sekali dari root repo ini
.\scripts\install_start_shortcut.ps1
```
Setelah itu, di terminal PowerShell BARU manapun (yang otomatis memuat `$PROFILE`), cukup:
```powershell
cd "path\ke\MyStocks"
start
```
Menjalankannya lagi saat semuanya sudah jalan itu aman (idempotent) -- terdeteksi otomatis dan dilewati, bukan dijalankan dobel.

**Scheduler mengejar hari yang terlewat**: kalau PC dimatikan/tidur saat jadwal harian (16:30 WIB) seharusnya jalan, `scripts/scheduler_loop.py` otomatis menjalankan catch-up begitu dinyalakan lagi (bukan menunggu jadwal besok) -- ditemukan & diperbaiki setelah kejadian nyata: data sempat basi 4 hari tanpa ada yang sadar karena laptop dev tidak menyala terus-menerus seperti VPS production.

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

Butuh minimal 60 hari price_history per ticker (indikator seperti ATR/RSI butuh window minimum; ticker yang lebih baru dari itu otomatis di-skip dengan warning, bukan error). Base indikator (RSI/MACD/Bollinger/ATR/CMF/MFI/OBV) memakai library `ta`.

Termasuk klasifikasi **regime** berbasis aturan (`features/regime.py`: overextended/bearish/bottoming/early_reversal/bullish/accumulation/sideways), dipakai sebagai fitur oleh kedua model DAN sebagai basis kandidat Turnaround serta Momentum Screener.

## ML Research (Fase 3, di Google Colab)

Colab tidak bisa menjangkau database lokal/dev, jadi datanya diekspor dulu ke Parquet:

```bash
python -m scripts.export_for_colab
```

Menghasilkan `data/export_for_colab_features.parquet` dan `..._prices.parquet` — dipakai baik di Colab (notebook) maupun langsung oleh skrip training/backtest di `scripts/` (lihat "Riset & Backtest" di bawah — banyak validasi terbaru dikerjakan begini, bukan lewat notebook).

## Model Swing (10 hari)

Skor tiap ticker pakai model **`direction_xgboost_v5`** (42 fitur, walk-forward validated), hasilkan keputusan BUY/WATCH/AVOID + Entry/SL/TP:

```bash
python -m engine.predict
```

Aturan keputusan (`engine/decision.py`, histori tuning lengkap ada di docstring file itu):
- Entry/SL/TP diturunkan dari `target_pct`/`stop_pct` yang sama dengan definisi label (5% / 2.5%, R:R 2.0).
- **BUY**: probability ≥ **60%** (`BUY_THRESHOLD`, sengaja diturunkan dari 65% hasil tuning walk-forward, atas permintaan eksplisit supaya sinyal BUY tidak kosong berhari-hari berturut-turut). **WATCH**: probability ≥ base rate historis (~30%) tapi < BUY_THRESHOLD. **AVOID**: di bawah base rate, atau harga ≤ Rp50 (gocap floor, tick-size mendominasi sinyal di harga sekecil itu).
- Precision walk-forward tercatat 84.5% di threshold 65% (headline lama); di threshold live 60% precision sesungguhnya sekitar 76-78% (dicek langsung, bukan diasumsikan) — kedua angka ditampilkan berdampingan di halaman **Info Model** supaya tidak membingungkan.

## Model Turnaround (6 bulan)

Skor ticker yang **sedang** bearish/bottoming untuk potensi berbalik arah:

```bash
python -m engine.predict_turnaround
```

Model **`turnaround_xgboost_v1`** (37 fitur) — POTENSIAL kalau probabilitas ≥85% untuk mencapai regime early_reversal/bullish dan **bertahan** di sana ≥20 hari perdagangan dalam 6 bulan ke depan, tanpa jatuh lagi ke bearish/bottoming/overextended. Base rate historis sudah tinggi (~82%) karena kebanyakan saham bearish/bottoming memang akhirnya membaik — nilainya lebih ke menyisihkan ~15% yang kemungkinan besar GAGAL, bukan menemukan yang pasti berhasil. Precision walk-forward ~92.4%.

Definisi label & 3 kali kalibrasi (v1 terlalu longgar, v2 terlalu ketat, v3 titik tengah yang dipakai) didokumentasikan lengkap di `scripts/turnaround_labels.py`.

## Momentum Screener (aturan teknikal, BUKAN model ML)

Screener berbasis RSI/MACD/volume/money-flow yang **ditentukan & divalidasi manual** (bukan dilatih dari data), untuk eksplorasi teknikal langsung di aplikasi (`features/momentum_screener.py`). Tidak ada perintah CLI terpisah -- dihitung on-demand tiap halaman **Momentum Screener** dibuka.

**Satu kombinasi Tervalidasi** ditemukan lewat backtest 5 tahun + grid search sistematis (`scripts/backtest_momentum_screener.py`, `scripts/search_momentum_rules.py`, `scripts/grid_search_momentum_rules.py`): regime **bottoming** + momentum histogram menguat + money flow negatif (distribusi, bukan akumulasi -- kounter-intuitif tapi konsisten: saham yang masih terlihat lemah di permukaan justru punya ruang lebih besar untuk mengejutkan naik) + volume relatif ≥0.8x. Win rate **39.8%** vs baseline acak 30.6% (target sama seperti Swing: naik ≥5% sebelum -2.5% dalam 10 hari) -- terbukti lebih baik dari acak secara statistik (Wilson 95% lower bound 36.7%), tapi jauh di bawah precision Swing/Turnaround. Semua kriteria LAIN di halaman ini (divergence, regime priority lama, RSI pivot distance) tetap murni heuristik yang masuk akal tapi belum terbukti -- dilabeli jelas di aplikasi.

## Rekomendasi Emitten (gabungan lintas-alat)

Halaman yang menyaring & menggabungkan hasil Swing + Turnaround + Momentum Screener sekaligus -- tidak ada perhitungan baru, murni agregasi. Backtest (`scripts/backtest_triple_intersection.py`) menunjukkan saham yang lolos ketiganya sekaligus (Swing ≥WATCH, Turnaround POTENSIAL, Momentum Tervalidasi) punya win rate lebih tinggi (42.3%, n=435) daripada Momentum Screener sendirian (39.8%) -- dengan catatan metodologi: pengujian gabungan ini menjalankan model Swing/Turnaround yang sudah terlatih dari SELURUH histori ke tanggal masa lalu, jadi anggap sebagai indikasi penguat, bukan bukti seketat walk-forward validation asli.

## Riset & Backtest

Skrip di `scripts/` yang bukan bagian dari pipeline harian, tapi mendokumentasikan validasi empiris di balik keputusan desain -- semuanya dijalankan dan hasilnya nyata, bukan asumsi:

- `test_feature_pruning.py`, `test_foreign_flow_feature.py`, `test_macd_zscore_feature.py` — apakah fitur tertentu benar-benar menambah akurasi model (kadang tidak, meski "terasa" seharusnya membantu).
- `tune_v5.py`, `tune_turnaround.py` — grid search hyperparameter + threshold sweep.
- `turnaround_labels.py` — 3 kalibrasi definisi label Turnaround.
- `test_turnaround_candidate_scope.py` — apakah mempersempit kandidat Turnaround membantu (tidak).
- `backtest_momentum_screener.py`, `search_momentum_rules.py`, `grid_search_momentum_rules.py`, `backtest_triple_intersection.py` — pencarian & validasi kombinasi aturan Momentum Screener + gabungan lintas-alat.

## Prediction & Decision Engine

Hasil Swing dan Turnaround tersimpan di tabel `predictions` yang sama (dibedakan `model_version`, upsert per `stock_code`+`date`+`model_version`, aman dijalankan berulang). Ticker tanpa `feature_daily` atau dengan fitur yang hilang di-skip dengan warning jelas, bukan crash atau prediksi dari data rusak.

## Aplikasi Streamlit

```bash
streamlit run app/Home.py
```

- **Home** — dashboard: pencarian cepat langsung ke Detail Saham, kartu ringkas 3 mode screening (Swing/Turnaround, Long-term Investment masih "Segera Hadir").
- **Detail Saham** — chart 6 panel (Harga, Volume, RSI, MACD, CMF, Foreign Flow) dengan crosshair lintas-panel, harga live, panel sektor/industri & fundamental, prediksi Swing dengan konteks base rate (supaya angka WATCH ~30% tidak disalahartikan sebagai model gagal).
- **Swing** — screener utama (ranking probabilitas, filter sidebar, live price overlay, tombol update harga).
- **Turnaround** — screener kandidat bearish/bottoming yang berpotensi berbalik arah.
- **Momentum Screener** — filter manual RSI/MACD/volume/CMF dengan kategori "✅ Sinyal Tervalidasi" yang disorot terpisah dari kriteria heuristik lainnya.
- **Rekomendasi Emitten** — gabungan lintas-alat, saham yang disepakati lebih dari satu screener sekaligus.
- **Info Model** — detail kedua model ML (Swing & Turnaround): umur model, pengingat retrain manual, precision walk-forward vs threshold live, daftar fitur.

Tema warna diatur di `.streamlit/config.toml` (dark mode) + `app/style.py` (badge/kartu). `/app` murni presentation layer — hanya query database & baca `models/*_metadata.json`, tidak ada logic pipeline/training di dalamnya.

## Production Deployment

Satu VPS menjalankan semuanya lewat Docker Compose: MySQL (self-host, volume persisten), aplikasi Streamlit, dan scheduler yang menjalankan `ingest_price → build_features → predict → predict_turnaround → monitor` otomatis tiap hari jam 16:30 WIB (bisa diubah via `SCHEDULER_RUN_HOUR`/`SCHEDULER_RUN_MINUTE`). Scheduler otomatis catch-up kalau ada hari yang terlewat (lihat "Windows tanpa Docker" di atas -- logika yang sama berlaku di semua environment).

**Rekomendasi hosting gratis:** [Oracle Cloud "Always Free"](https://www.oracle.com/cloud/free/) (VM Ampere A1, 2 OCPU/12GB RAM, gratis selamanya bukan trial) — cukup untuk stack ini.

### Setup di VPS

```bash
git clone https://github.com/hpwiyoto/MyStocks.git
cd MyStocks
cp .env.example .env   # isi kredensial MySQL sungguhan + MYSQL_ROOT_PASSWORD
                        # TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID opsional untuk alert
docker compose up -d --build
```

Verifikasi:
```bash
docker compose ps                                  # ketiganya harus "Up" / mysql "healthy"
docker exec mystocks-scheduler-1 python -m scripts.run_daily   # trigger manual, cek log
curl -I http://localhost:8501                       # HTTP 200
```

Buka `http://<IP-VPS>:8501` di browser. Untuk domain/HTTPS, pasang reverse proxy (nginx/caddy) di depan port 8501 — di luar scope ini.

**Monitoring:** `scripts/monitor.py` mendeteksi dua jenis kegagalan — gagal ingest eksplisit, dan data "diam-diam basi" (fetch sukses tapi tanggal terbaru tidak maju >4 hari). Tanpa `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`, alert cuma masuk log (`data/logs/pipeline.log` di dalam container `scheduler`); isi keduanya untuk dapat notifikasi Telegram juga.
