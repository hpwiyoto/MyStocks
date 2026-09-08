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
- **BUY**: probability ≥ **60%** (`BUY_THRESHOLD`, sengaja diturunkan dari 65% hasil tuning walk-forward, atas permintaan eksplisit supaya sinyal BUY tidak kosong berhari-hari berturut-turut) -- **kecuali saat IHSG sendiri sedang turun** (return 20 hari negatif), di mana ambangnya naik ke **65%** (`IHSG_DECLINE_BUY_THRESHOLD`, dicek sekali per hari lewat `pipeline.yfinance_source.is_ihsg_declining()`, bukan per saham). **WATCH**: probability ≥ base rate historis (~30%) tapi < ambang BUY yang berlaku hari itu. **AVOID**: di bawah base rate, atau harga ≤ Rp50 (gocap floor, tick-size mendominasi sinyal di harga sekecil itu).
- Precision walk-forward tercatat 84.5% di threshold 65% (headline lama); di threshold live 60% precision sesungguhnya sekitar 76-78% (dicek langsung, bukan diasumsikan) — kedua angka ditampilkan berdampingan di halaman **Info Model** supaya tidak membingungkan.
- **Angka precision itu RATA-RATA 4 fold walk-forward, bukan angka tunggal yang stabil** — dicek per-fold (`scripts/check_fold_drift.py`) dan ternyata bervariasi 72-88% antar fold, dengan fold PALING BARU (Mar-Agu 2026) justru yang PALING LEMAH (71-72%, sinyal BUY paling jarang). Berkorelasi nyata dengan tren IHSG sendiri (fold lemah = periode IHSG sedang turun) -- tapi menambahkan fitur tren IHSG sebagai perbaikan justru terbukti memperparah drastis, bukan membantu (`scripts/test_ihsg_regime_feature.py`, lihat docstring-nya untuk detail kenapa). Model masih perlu dipantau berkala untuk drift, bukan dianggap "sudah pasti bagus selamanya" hanya dari validasi sekali di 27 Agustus.
- **Perbaikan yang TERBUKTI membantu** (`scripts/test_regime_conditional_threshold.py`): bukan menambah fitur, tapi menaikkan ambang keputusan BUY saat IHSG turun (lihat aturan keputusan di atas). Pooled walk-forward: sinyal BUY saat IHSG turun tadinya menang 71,6% (Wilson LB 65,9%) vs 75,2% (LB 72,5%) saat IHSG tidak turun -- setelah ambang dinaikkan khusus untuk kondisi itu, win rate gabungan naik ke 75,7% (LB 73,2%) sambil tetap mempertahankan 93% volume sinyal. Sudah live di `engine/decision.py`, bukan cuma catatan riset.

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
- `check_fold_drift.py` — apakah performa Swing model stabil dari waktu ke waktu, dicek per walk-forward fold (bukan dirata-rata) plus base rate mentah per tahun kalender. Menemukan performa TIDAK stabil, fold terbaru paling lemah.
- `test_ihsg_regime_feature.py` — mengikuti temuan `check_fold_drift.py`: apakah menambahkan fitur tren IHSG (indeks, bukan per-saham) memperbaiki fold yang lemah tadi. Terbukti memperparah drastis, tidak diadopsi.
- `test_regime_conditional_threshold.py` — versi yang benar dari ide yang sama (terinspirasi paper eksternal soal model per-rezim pasar): bukan fitur training, tapi ambang keputusan yang beda saat IHSG turun. Terbukti membantu sungguhan (lihat "Model Swing" di bawah) -- sudah diadopsi ke `engine/decision.py`.
- `backtest_momentum_screener.py`, `search_momentum_rules.py`, `grid_search_momentum_rules.py`, `backtest_triple_intersection.py` — pencarian & validasi kombinasi aturan Momentum Screener + gabungan lintas-alat.

## Prediction & Decision Engine

Hasil Swing dan Turnaround tersimpan di tabel `predictions` yang sama (dibedakan `model_version`, upsert per `stock_code`+`date`+`model_version`, aman dijalankan berulang). Ticker tanpa `feature_daily` atau dengan fitur yang hilang di-skip dengan warning jelas, bukan crash atau prediksi dari data rusak.

## Aplikasi Streamlit

```bash
streamlit run app/Home.py
```

- **Home** — dashboard: pencarian cepat langsung ke Detail Saham, kartu ringkas untuk semua mode screening (Swing, Turnaround, Momentum Screener, Rekomendasi Emitten; Long-term Investment masih "Segera Hadir").
- **Detail Saham** — chart 6 panel (Harga, Volume, RSI, MACD, CMF, Foreign Flow) dengan crosshair lintas-panel plus label nilai tiap panel saat hover, harga live, panel sektor/industri & fundamental, prediksi Swing dengan konteks base rate (supaya angka WATCH ~30% tidak disalahartikan sebagai model gagal).
- **Swing** — screener utama (ranking probabilitas, filter sidebar, live price overlay, tombol update harga).
- **Turnaround** — screener kandidat bearish/bottoming yang berpotensi berbalik arah.
- **Momentum Screener** — filter manual RSI/MACD/volume/CMF dengan kategori "✅ Sinyal Tervalidasi" yang disorot terpisah dari kriteria heuristik lainnya.
- **Rekomendasi Emitten** — gabungan lintas-alat, saham yang disepakati lebih dari satu screener sekaligus.
- **Info Model** — detail kedua model ML (Swing & Turnaround): umur model, pengingat retrain manual, precision walk-forward vs threshold live, daftar fitur.
- **Admin** — hanya untuk admin (lihat "Login & Approval Akses" di bawah): approve/reject permintaan akses baru.

Tema warna diatur di `.streamlit/config.toml` (dark mode) + `app/style.py` (badge/kartu). `/app` murni presentation layer — hanya query database & baca `models/*_metadata.json`, tidak ada logic pipeline/training di dalamnya.

## Login & Approval Akses

Aplikasi ini privat: pengunjung baru bisa melihat pratinjau **2 halaman** secara bebas (mode tamu, tanpa login -- perkenalan singkat sebelum diminta masuk), lalu diarahkan ke layar login Google. Setelah login, akun baru masuk status **"Menunggu Persetujuan"** dan tidak bisa memakai aplikasi sampai di-approve lewat halaman **Admin** (hanya bisa dibuka oleh `heru.purbowiyoto@gmail.com`, dikonfigurasi via `app/auth.py`'s `ADMIN_EMAIL`) -- akun admin sendiri otomatis ter-approve begitu pertama kali login, supaya tidak ada ayam-telur (harus ada admin untuk approve admin pertama).

Implementasinya memakai fitur bawaan Streamlit (`st.login`/`st.user`, OpenID Connect) -- bukan library pihak ketiga -- ditambah satu tabel `app_users` (`app/db.py`: email, nama, status pending/approved/rejected) dan satu email notifikasi (Gmail SMTP, `app/email_notify.py`) tiap ada pendaftar baru.

**Setup (sekali saja, per environment -- dev lokal dan VPS punya redirect URI yang beda):**

1. **Google OAuth Client ID** (untuk tombol "Login dengan Google"):
   - Buka [Google Cloud Console](https://console.cloud.google.com/) → buat project baru (atau pakai yang sudah ada) → **APIs & Services → OAuth consent screen** → isi minimal (nama app, email support) → **User Type: External** (cukup untuk pemakaian pribadi, tidak perlu publish/verifikasi kalau hanya dipakai beberapa orang yang sudah dikenal).
   - **APIs & Services → Credentials → Create Credentials → OAuth client ID** → Application type **Web application**.
   - **Authorized redirect URIs**, tambahkan (harus persis, termasuk skema & port):
     - Dev lokal: `http://localhost:8501/oauth2callback`
     - VPS/production: `http://<IP-atau-domain-VPS>:8501/oauth2callback` (ganti sesuai alamat sungguhan; kalau nanti pasang HTTPS di depan reverse proxy, ini juga berubah ke `https://...`)
   - Simpan **Client ID** dan **Client Secret** yang muncul.

2. **Gmail App Password** (untuk mengirim email notifikasi pendaftar baru):
   - Prasyarat: verifikasi 2 langkah (2FA) harus aktif di akun Gmail pengirim.
   - Buka [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords) → buat App Password baru (nama bebas, misal "MyStocks") → salin 16 karakter yang muncul (ini BUKAN password Gmail biasa -- password Gmail biasa akan DITOLAK oleh SMTP Google).

3. **Isi `secrets.toml`**: salin `.streamlit/secrets.toml.example` → `.streamlit/secrets.toml` (file ini sengaja di-gitignore, tidak pernah masuk git), lalu isi `client_id`/`client_secret`/`redirect_uri` dari langkah 1, `app_password` dari langkah 2, dan `cookie_secret` (string acak apa saja -- generate dengan `python -c "import secrets; print(secrets.token_hex(32))"`).

4. **Jalankan seperti biasa** (`streamlit run app/Home.py` atau `start` di Windows) -- tombol "Login dengan Google" otomatis aktif begitu `secrets.toml` lengkap. Login pertama dengan `heru.purbowiyoto@gmail.com` langsung ter-approve; akun lain masuk antrian "Menunggu Persetujuan" dan admin akan menerima email untuk membuka halaman **Admin** dan approve/reject.

**Produksi (VPS/Docker)**: `docker-compose.yml`'s service `app` me-mount `.streamlit/secrets.toml` sebagai read-only volume saat runtime (bukan di-build ke image -- `.dockerignore` sengaja mengecualikannya supaya secret tidak pernah ikut ter-bake ke layer image yang mungkin di-push/dibagikan). Buat file itu langsung di VPS (isi redirect URI versi VPS-nya), bukan lewat git.

## Production Deployment

Satu VPS menjalankan semuanya lewat Docker Compose. Scheduler menjalankan `ingest_price → build_features → predict → predict_turnaround → monitor` otomatis tiap hari jam 16:30 WIB (bisa diubah via `SCHEDULER_RUN_HOUR`/`SCHEDULER_RUN_MINUTE`), dengan catch-up otomatis kalau ada hari yang terlewat (lihat "Windows tanpa Docker" di atas -- logika yang sama berlaku di semua environment).

**Sejak ada Login Google, HTTPS + domain asli WAJIB** (bukan lagi opsional) -- Google menolak redirect URI berbasis HTTP kecuali untuk `localhost`. Dua opsi VPS di bawah ini keduanya sudah termasuk HTTPS otomatis.

### Opsi A -- VM besar (Oracle Cloud Always Free), MySQL

[Oracle Cloud "Always Free"](https://www.oracle.com/cloud/free/) (VM Ampere A1, 2 OCPU/12GB RAM, gratis selamanya) -- kalau dapat slot (sering kehabisan kapasitas per region, coba region lain kalau gagal). Pakai `docker-compose.yml` (MySQL self-host, volume persisten) apa adanya:

```bash
git clone https://github.com/hpwiyoto/MyStocks.git
cd MyStocks
cp .env.example .env   # isi kredensial MySQL sungguhan + MYSQL_ROOT_PASSWORD
                        # TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID opsional untuk alert
cp Caddyfile.example Caddyfile   # ganti domain placeholder dengan domain/sslip.io Anda
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # isi kredensial Google OAuth + Gmail
docker compose up -d --build
```

### Opsi B -- VM kecil gratis-selamanya (Google Cloud Always Free `e2-micro`), SQLite

Dipakai untuk pemakaian pribadi/skala kecil (RAM 1GB, tidak cukup untuk MySQL sebagai container terpisah). Pakai `docker-compose.lite.yml` (tanpa service MySQL -- otomatis jatuh ke SQLite, sama seperti dev lokal; lihat komentar di file itu untuk detail).

**1. Buat VM** (region harus salah satu yang Always-Free eligible: `us-west1`, `us-central1`, atau `us-east1`):
- Google Cloud Console → **Compute Engine → VM instances → Create Instance**
- Machine type: `e2-micro`
- Boot disk: Ubuntu (versi LTS terbaru), 30GB standard persistent disk (batas gratis)
- Firewall: centang **Allow HTTP traffic** dan **Allow HTTPS traffic**
- Setelah dibuat, **reserve IP eksternalnya jadi Static** (VPC Network → IP addresses → ubah dari Ephemeral ke Static) -- supaya IP tidak berubah tiap VM restart. Static IP tetap gratis selama terpasang ke VM yang menyala.

**2. Siapkan domain gratis dari IP itu** -- tidak perlu beli domain atau setup DNS: `sslip.io` otomatis mengubah IP jadi hostname valid. IP `34.123.45.67` → domain `34-123-45-67.sslip.io` (ganti titik dengan strip).

**3. Update Google Cloud OAuth Client** (Console → Google Auth Platform → Clients → client Anda) -- tambahkan Authorized redirect URI: `https://<domain-sslip-Anda>/oauth2callback`.

**4. SSH ke VM, install Docker, tambah swap** (RAM 1GB perlu bantuan disk untuk lonjakan sesaat -- mencegah crash, bukan solusi kecepatan):
```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

**5. Clone & deploy:**
```bash
git clone https://github.com/hpwiyoto/MyStocks.git
cd MyStocks
cp .env.example .env   # MYSQL_* dibiarkan kosong/dihapus -- tidak dipakai di opsi ini
                        # TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID opsional untuk alert
cp Caddyfile.example Caddyfile
nano Caddyfile          # ganti domain placeholder dengan domain sslip.io dari langkah 2
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
nano .streamlit/secrets.toml   # isi client_id/client_secret dari langkah 3, redirect_uri =
                                # "https://<domain-sslip-Anda>/oauth2callback", cookie_secret
                                # acak, dan app_password Gmail (opsional, boleh menyusul)
docker compose -f docker-compose.lite.yml up -d --build
```

### Verifikasi (kedua opsi)

```bash
docker compose ps                                              # (opsi A) atau tambahkan -f docker-compose.lite.yml (opsi B)
docker exec mystocks-scheduler-1 python -m scripts.run_daily   # trigger manual, cek log
curl -I https://<domain-Anda>                                   # HTTP 200, sertifikat HTTPS otomatis dari Caddy
```

Buka `https://<domain-Anda>` di browser (bukan `http://` atau IP polos -- itu yang dipakai Google untuk redirect setelah login).

**Monitoring:** `scripts/monitor.py` mendeteksi dua jenis kegagalan — gagal ingest eksplisit, dan data "diam-diam basi" (fetch sukses tapi tanggal terbaru tidak maju >4 hari). Tanpa `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`, alert cuma masuk log (`data/logs/pipeline.log` di dalam container `scheduler`); isi keduanya untuk dapat notifikasi Telegram juga.
