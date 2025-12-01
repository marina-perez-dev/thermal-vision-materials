# Lance format/lint/hooks sur tout le dépôt (utilise .venv si présent)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

$repoRoot = Split-Path -Parent $PSScriptRoot
$blackExe = Join-Path $repoRoot '.venv\Scripts\black.exe'
$ruffExe = Join-Path $repoRoot '.venv\Scripts\ruff.exe'
$preCommitExe = Join-Path $repoRoot '.venv\Scripts\pre-commit.exe'

# Format with Black
if (Test-Path $blackExe) {
    & $blackExe $repoRoot
} else {
    Write-Warning "black not found in .venv, trying system black"
    black .
}

# Ruff fix (or check)
if (Test-Path $ruffExe) {
    & $ruffExe --fix $repoRoot
} else {
    Write-Warning "ruff not found in .venv, trying system ruff"
    ruff --fix .
}

# Run pre-commit hooks on all files
if (Test-Path $preCommitExe) {
    & $preCommitExe run --all-files
} else {
    pre-commit run --all-files
}

Write-Output "Checks complete."