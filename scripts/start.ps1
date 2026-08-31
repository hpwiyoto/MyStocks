# One-command native-Windows bring-up (no Docker): ensures the venv exists
# and dependencies are installed, launches the scheduler + Streamlit app
# each in their own visible PowerShell window (so logs stay visible and
# either can be closed/Ctrl+C'd independently, same idea as `docker compose
# logs` per-service), waits for the app to actually answer, then opens the
# browser. Meant to be run as the bare `start` command -- see the `start`
# function installed into $PROFILE by scripts\install_start_shortcut.ps1,
# which special-cases the alias only while the cwd is inside this repo.
#
# Database is local SQLite (pipeline/db.py, zero-config) -- unlike the
# Codespaces/VPS setup (scripts/start.sh, docker-compose.yml), there's no
# MySQL/dockerd bring-up step needed here at all.

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "== MyStocks: starting (native, no Docker) ==" -ForegroundColor Cyan

# --- 1. venv: create once, reused on every subsequent `start`. ---
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "-> membuat virtual environment (.venv)..."
    py -m venv .venv
    if (-not (Test-Path $VenvPython)) {
        Write-Host "!! gagal membuat .venv -- pastikan 'py' (Python launcher) terpasang." -ForegroundColor Red
        exit 1
    }
}

Write-Host "-> memastikan dependencies terpasang (pip install, cepat kalau sudah lengkap)..."
& $VenvPython -m pip install --quiet --disable-pip-version-check -r (Join-Path $RepoRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    Write-Host "!! pip install gagal -- lihat error di atas." -ForegroundColor Red
    exit 1
}

# --- 2. Scheduler: skip if one's already running (avoid duplicate daily
# jobs), matching docker-compose's restart:unless-stopped idempotency. ---
$schedulerRunning = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -like "*scripts.scheduler_loop*" }
if ($schedulerRunning) {
    Write-Host "-> scheduler sudah berjalan (PID $($schedulerRunning.ProcessId)), tidak dijalankan ulang"
} else {
    Write-Host "-> menjalankan scheduler di jendela baru..."
    Start-Process powershell -ArgumentList @(
        "-NoExit", "-Command",
        "`$host.UI.RawUI.WindowTitle = 'MyStocks - Scheduler'; Set-Location '$RepoRoot'; & '$VenvPython' -m scripts.scheduler_loop"
    )
}

# --- 3. Streamlit app: skip (re)launch if something's already answering on
# :8501 -- avoids a confusing "address already in use" from a second copy. ---
function Test-Port8501 {
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $client.Connect("127.0.0.1", 8501)
        $client.Close()
        return $true
    } catch {
        return $false
    }
}

if (Test-Port8501) {
    Write-Host "-> ada yang sudah jalan di :8501, tidak dijalankan ulang"
} else {
    Write-Host "-> menjalankan aplikasi Streamlit di jendela baru..."
    Start-Process powershell -ArgumentList @(
        "-NoExit", "-Command",
        "`$host.UI.RawUI.WindowTitle = 'MyStocks - App'; Set-Location '$RepoRoot'; & '$VenvPython' -m streamlit run app/Home.py --server.port=8501 --server.address=0.0.0.0"
    )
}

# --- 4. Wait for it to actually answer, not just "process started". ---
Write-Host -NoNewline "-> menunggu Streamlit siap di :8501"
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    if (Test-Port8501) { $ready = $true; break }
    Write-Host -NoNewline "."
    Start-Sleep -Seconds 2
}
Write-Host ""

if ($ready) {
    Write-Host "== Siap: http://localhost:8501 ==" -ForegroundColor Green
    Start-Process "http://localhost:8501"
} else {
    Write-Host "!! Streamlit belum merespon setelah 60 detik -- cek jendela 'MyStocks - App' untuk error." -ForegroundColor Red
    exit 1
}
