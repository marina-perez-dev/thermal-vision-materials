# Script d'installation des hooks pre-commit (exécuter depuis la racine du dépôt)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

$repoRoot = Split-Path -Parent $PSScriptRoot
$pipPath = Join-Path $repoRoot '.venv\Scripts\pip.exe'
$preCommitPath = Join-Path $repoRoot '.venv\Scripts\pre-commit.exe'

if (Test-Path $pipPath) {
    Write-Output "Using venv pip: $pipPath"
    & $pipPath install --upgrade pip
    & $pipPath install pre-commit
} else {
    Write-Warning ".venv pip not found, using system pip"
    pip install --user --upgrade pip
    pip install --user pre-commit
}

# installer les hooks dans .git/hooks
if (Test-Path $preCommitPath) {
    Write-Output "Installing pre-commit hooks using: $preCommitPath"
    & $preCommitPath install
} else {
    Write-Output "Running 'pre-commit install' (system)"
    pre-commit install
}

Write-Output "pre-commit hooks installed. Commit .pre-commit-config.yaml to share config."