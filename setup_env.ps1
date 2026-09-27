# =============================================================================
# FedMed — One-Time Environment Setup Script
# =============================================================================
# Run this ONCE on your machine. After that, all FedMed code works automatically.
#
# HOW TO USE:
#   1. Edit the two paths below to match where YOU stored the files on your machine.
#   2. Open PowerShell and run:  .\setup_env.ps1
#   3. Restart your terminal / VS Code.
#   4. Done. Never touch this again.
#
# DO NOT commit your edited version to Git.
# Everyone has different paths — keep your local edits local.
# =============================================================================

# ── EDIT THESE TWO LINES ONLY ────────────────────────────────────────────────

# Path to the data_prep folder (contains client_1.parquet, train.parquet, etc.)
$DATA_DIR = "R:\VSCODE\Main_Project\data_prep"

# Path to the folder where the actual image files (PNG) are stored
$IMAGE_ROOT = "R:\FedMed_Data"

# ─────────────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "Setting FedMed environment variables permanently for current user..." -ForegroundColor Cyan

[System.Environment]::SetEnvironmentVariable("FEDMED_DATA_DIR",   $DATA_DIR,   "User")
[System.Environment]::SetEnvironmentVariable("FEDMED_IMAGE_ROOT", $IMAGE_ROOT, "User")

Write-Host ""
Write-Host "Done! Variables set:" -ForegroundColor Green
Write-Host "  FEDMED_DATA_DIR   = $DATA_DIR"
Write-Host "  FEDMED_IMAGE_ROOT = $IMAGE_ROOT"
Write-Host ""
Write-Host "Restart your terminal or VS Code for the changes to take effect." -ForegroundColor Yellow
Write-Host ""
