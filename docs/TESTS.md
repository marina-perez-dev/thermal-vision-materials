## Tests

1) Installer pytest (avec l'environnement activé) :
```powershell
python -m pip install pytest
# ou ajouter pytest à requirements.txt puis
python -m pip install -r requirements.txt
```

2) Lancer les tests localement :
```powershell
pytest -q
```

3) Vérifications sûres (PowerShell) :
```powershell
if (Test-Path .\tests) { pytest -q } else { Write-Output "Aucun dossier tests trouvé" }
```

4) Exemple de test basique (mocks) pour thermal_sensors
- Crée `tests/test_thermal_sensors_basic.py` et adapte l'import selon ton module.

## Intégration continue (exemple GitHub Actions)
- Assure-toi que la CI exécute `pytest -q`. Exemple de workflow dans `.github/workflows/pytest.yml`.
- Si tu utilises un autre CI, adapte la commande `pytest -q` dans le job de test.