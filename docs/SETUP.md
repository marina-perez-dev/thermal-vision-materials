# Setup (environnement de développement)

## Prérequis
- Python 3.8+ installé et accessible depuis le PATH.
- Git (si vous utilisez pre-commit ou les hooks).

## 1) Créer et activer le virtualenv
Depuis la racine du projet :
```powershell
# créer le venv
python -m venv .venv
```

Activation (PowerShell recommandé) :
```powershell
# dot-source pour que l'activation affecte la session courante
. .\scripts\activate.ps1

# ou activer directement le venv
. .\.venv\Scripts\Activate.ps1
```

Autres options d'activation :
- CMD : `.venv\Scripts\activate.bat`
- Git Bash / WSL : `source .venv/Scripts/activate`

> Remarque : utilisez le dot-sourcing (`. .\scripts\activate.ps1`) si vous voulez que le script modifie la session PowerShell courante.

## 2) Installer les dépendances du projet
Avec l'environnement activé :
```powershell
pip install -r requirements.txt
```
Si `requirements.txt` n'existe pas, installez les paquets souhaités puis générez le fichier (voir section suivante).

## 3) Mettre à jour `requirements.txt`
```powershell
pip freeze > requirements.txt
```

## 4) Outils de développement (format/lint/hooks)
Installer les outils recommandés (avec l'env activé) :
```powershell
python -m pip install --upgrade pip
pip install black ruff pre-commit
```

## 5) Installer les hooks pre-commit
Option A — script fourni (depuis la racine du dépôt) :
```powershell
.\scripts\install-hooks.ps1
```

Option B — installer manuellement :
```powershell
.\.venv\Scripts\pre-commit.exe install
# ou si pre-commit est dans le PATH
pre-commit install
```

- Le fichier `.pre-commit-config.yaml` doit être placé à la racine du dépôt et commité.
- Les hooks s'exécutent automatiquement lors d'un `git commit`.

## 6) Commandes utilitaires
```powershell
# exécuter tous les checks et corrections auto (script fourni)
.\scripts\check-all.ps1

# lancer les hooks sur tous les fichiers
.\.venv\Scripts\pre-commit.exe run --all-files

# formater tout le projet
black .

# vérifier (lint) sans modifier
ruff check .

# appliquer les corrections auto
ruff --fix .
```

## 7) VS Code
Sélectionner l'interpréteur : `${workspaceFolder}\.venv\Scripts\python.exe` (coin inférieur droit ou via "Python: Select Interpreter").

## Notes
- Le dossier `.venv` doit être ignoré par Git (vérifier `.gitignore`).
- Le script `scripts\activate.ps1` peut contourner temporairement la policy en session ; utilisez-le avec précaution.
- Si des étapes semblent manquantes pour votre flux (CI, containers, WSL), adaptez les commandes en conséquence.
