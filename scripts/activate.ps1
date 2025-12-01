# Script d'activation qui contourne temporairement la policy et source le Activate.ps1 du venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

$activatePath = Join-Path $PSScriptRoot '..\.venv\Scripts\Activate.ps1'
if (Test-Path $activatePath) {
    # dot-source pour que l'activation affecte la session courante
    . $activatePath
} else {
    Write-Error "Fichier d'activation introuvable : $activatePath"
}

python -m pip install --upgrade pip
pip install pre-commit