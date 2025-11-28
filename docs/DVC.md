# DVC — Checklist et commandes (projet)

Etat attendu
- .dvc/                      -> présent (dvc init)
- .dvc/config                -> remote configuré (local ou cloud)
- dvc.yaml                   -> pipeline ingest -> preprocess
- dvc.lock                   -> généré après `dvc repro`
- outputs (data/raw..., data/processed...) -> générés par `dvc repro`

Commandes utiles (PowerShell)
- Exécuter pipeline :
  dvc repro

- Pousser cache vers remote (backup) :
  dvc push -v

- Vérifier état remote vs local :
  dvc status -r <remote>

- Récupérer les outputs sur une autre machine :
  git clone <repo>
  dvc pull

Bonnes pratiques
- Committer dvc.yaml et dvc.lock pour la reproductibilité.
- Ne pas committer les fichiers volumineux suivis par DVC.
- Mettre les credentials cloud dans .dvc/config.local ou variables d’environnement.