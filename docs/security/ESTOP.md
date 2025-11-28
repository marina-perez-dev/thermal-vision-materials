# Documente le comportement de l'EStop (arrêt d'urgence)
## But
Décrire les conditions qui déclenchent l'EStop, les actions prises et la procédure de remise en service.

## Déclencheurs
- Appui sur bouton matériel d'arrêt d'urgence (physique).
- Watchdog capteur : perte des données au-delà d'un timeout (par défaut 1000 ms) ou valeurs invalides répétées.
- Commande logicielle explicite (/estop service/topic).

## Actions visibles
- Arrêt immédiat des actionneurs/moteurs (stop controllers).
- Mise en pause de la simulation Gazebo (optionnel).
- Publication d'un message sur `/estop_state` (std_msgs/Bool ou custom) avec reason et timestamp.
- Enregistrement d'un événement dans les logs/diagnostics.

## Interface ROS2 recommandée
- Topic: `/estop_state` (std_msgs/Bool + details via diagnostics).
- Service: `/reset_estop` (std_srvs/Trigger) pour réinitialiser après vérification.
- Paramètres:
  - `estop.hold = true/false` (bloque/release).
  - `estop.auto_on_watchdog = true` (activer arrêts automatiques).

## Procédure de remise en service
1. Vérifier l'origine de l'EStop (logs, diagnostics).
2. Corriger la cause (par ex. reconnecter capteur).
3. Exécuter les contrôles de sécurité manuels.
4. Appeler `/reset_estop` depuis un opérateur autorisé.
5. Re-valider via tests de santé (joint limits, capteurs OK) avant reprise.

## Tests & validation
- Scénarios unitaires: simuler perte de topic capteur > timeout -> vérifier publication `/estop_state` et controllers stoppés.
- Tests d'intégration: bouton matériel -> vérification arrêt sur systeme réel/simulé.