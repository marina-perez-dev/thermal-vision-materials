# Politique de gestion des secrets (ne pas committer credentials)

Principes généraux :
- Ne jamais committer de credentials, clés API, mots de passe ou certificats dans le dépôt.
- Mettre les variables sensibles dans des variables d'environnement, stores de secrets CI (ex: GitHub Actions secrets) ou services dédiés (Vault, AWS Secrets Manager).
- Garder un fichier `.env.example` avec les noms des variables mais sans valeurs réelles.

Fichiers & outils recommandés :
- `.gitignore` doit contenir: `.env`, `.secrets`, `credentials.*` (déjà présent).
- Ajouter un hook pre-commit (detect-secrets / git-secrets).
- Utiliser DVC correctement : ne pas stocker credentials DVC dans repo; configurer DVC remote via commandes `dvc remote modify` sur la CI avec secrets injectés.

Exemples de variables à stocker en externe (placeholders) :
- DVC_REMOTE_URL
- DVC_REMOTE_USER
- DVC_REMOTE_PASSWORD
- HIKMICRO_API_KEY
- MLX90640_TOKEN
- BOSON_KEY

CI / GitHub Actions :
- Définir secrets en Settings > Secrets et les référencer via `${{ secrets.DVC_REMOTE_PASSWORD }}`.
- Ne pas écrire secrets en clair dans les logs.

Rotation / révocation :
- Avoir procédure de rotation (changer clé, révoquer anciens tokens).
- Tenir un journal des changements quand une clé est compromise.

Détection / audit :
- Exécuter `pre-commit` et `detect-secrets` localement et en CI.
- En cas de fuite détectée, suivre la checklist d'urgence : révoquer la clé, remplacer, notifier parties prenantes.

Liens utiles :
- https://github.com/Yelp/detect-secrets
- DVC docs sur remote auth (https://dvc.org/doc)