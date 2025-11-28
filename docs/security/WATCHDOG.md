# Watchdog des capteurs : spécification et comportements
## Objectif
Surveiller la disponibilité et la validité des flux capteurs (Hikmicro, MLX90640, Boson ou stream simulé depuis dataset) et déclencher des actions (alerte / EStop / fallback).

## Capteurs optionnels supportés
- hikmicro
- mlx90640
- boson

## Mode simulation (avant obtention du matériel)
- Dataset : “Long‑wave Thermal Diurnal Material Classification” (CVPR 2025)
- Node simuleur publie sur `/sensor/<name>/image_raw` (sensor_msgs/Image) à 10 Hz.
- Nom du node simulation recommandé : `thermal_dataset_publisher`.

## Paramètres recommandés
- `watchdog.timeout_ms = 1000` (timeout pour heartbeat)
- `watchdog.max_missed = 3` (nombre de timeouts consécutifs avant action)
- `watchdog.invalid_thresholds = { min_C: -50, max_C: 500 }` (seuils de température plausibles)

## Comportement à la détection d'anomalie
- 1ère anomalie : log + publish diagnostics sur `/sensor_watchdog/<name>/status`.
- Après `max_missed` : si `watchdog.auto_estop=true` -> déclencher EStop (publier `/estop_state`).
- Option de fallback : basculer en mode "dataset-only" si hardware absent et simulation disponible.

## Interfaces ROS2 recommandées
- Topics:
  - `/sensor/<name>/image_raw` (sensor_msgs/Image)
  - `/sensor_watchdog/<name>/status` (diagnostic_msgs/DiagnosticStatus)
  - `/estop_state` (std_msgs/Bool ou message custom)
- Services:
  - `/reset_watchdog` (std_srvs/Trigger) — remet compteurs à zéro.
  - `/reset_estop` (std_srvs/Trigger)

## Tests
- Injecter messages manquants, valeurs hors-limites et vérifier transitions et logs.
- Valider que simulation (dataset) permet poursuite sans EStop si configuré.