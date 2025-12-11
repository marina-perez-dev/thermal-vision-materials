"""Module d'évaluation pour les modèles de classification thermique.

Ce module fournit des fonctions pour calculer et sauvegarder les métriques
d'évaluation complètes : accuracy, precision, recall, F1-score, et matrice
de confusion.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Calcule toutes les métriques d'évaluation.

    Args:
        y_true: Labels réels (one-hot encoded ou indices)
        y_pred: Prédictions (one-hot encoded ou indices)
        class_names: Liste des noms de classes (optionnel)

    Returns:
        Dictionnaire contenant toutes les métriques calculées
    """
    # Convertir en indices si nécessaire (one-hot -> indices)
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_indices = np.argmax(y_true, axis=1)
    else:
        y_true_indices = y_true.flatten().astype(int)

    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred_indices = np.argmax(y_pred, axis=1)
    else:
        y_pred_indices = y_pred.flatten().astype(int)

    # Calculer les métriques globales
    accuracy = accuracy_score(y_true_indices, y_pred_indices)

    # Calculer les métriques par classe
    precision_per_class = precision_score(
        y_true_indices, y_pred_indices, average=None, zero_division=0
    )
    recall_per_class = recall_score(
        y_true_indices, y_pred_indices, average=None, zero_division=0
    )
    f1_per_class = f1_score(
        y_true_indices, y_pred_indices, average=None, zero_division=0
    )

    # Calculer les métriques macro-averaged
    precision_macro = precision_score(
        y_true_indices, y_pred_indices, average="macro", zero_division=0
    )
    recall_macro = recall_score(
        y_true_indices, y_pred_indices, average="macro", zero_division=0
    )
    f1_macro = f1_score(
        y_true_indices, y_pred_indices, average="macro", zero_division=0
    )

    # Calculer la matrice de confusion
    cm = confusion_matrix(y_true_indices, y_pred_indices)

    # Préparer les résultats
    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_per_class": precision_per_class.tolist(),
        "recall_per_class": recall_per_class.tolist(),
        "f1_per_class": f1_per_class.tolist(),
        "confusion_matrix": cm.tolist(),
    }

    # Ajouter les noms de classes si fournis
    if class_names:
        metrics["class_names"] = class_names
        # Créer un dictionnaire par classe
        per_class_metrics = {}
        for i, class_name in enumerate(class_names):
            per_class_metrics[class_name] = {
                "precision": float(precision_per_class[i]),
                "recall": float(recall_per_class[i]),
                "f1_score": float(f1_per_class[i]),
            }
        metrics["per_class"] = per_class_metrics

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1-score (macro): {f1_macro:.4f}")
    logger.info(f"Precision (macro): {precision_macro:.4f}")
    logger.info(f"Recall (macro): {recall_macro:.4f}")

    return metrics


def save_confusion_matrix(
    confusion_matrix: np.ndarray,
    output_path: Path,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """Sauvegarde la matrice de confusion en tant qu'image PNG.

    Args:
        confusion_matrix: Matrice de confusion (2D numpy array)
        output_path: Chemin de sortie pour l'image
        class_names: Liste des noms de classes (optionnel)
        figsize: Taille de la figure (largeur, hauteur)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Normaliser la matrice pour l'affichage en pourcentages
    cm_normalized = confusion_matrix.astype("float") / (
        confusion_matrix.sum(axis=1)[:, np.newaxis] + 1e-8
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Créer l'image de la matrice de confusion
    im = ax.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Configurer les axes
    num_classes = confusion_matrix.shape[0]
    if class_names:
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
    else:
        tick_marks = np.arange(num_classes)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels([f"Class {i}" for i in range(num_classes)], rotation=45, ha="right")
        ax.set_yticklabels([f"Class {i}" for i in range(num_classes)])

    # Ajouter les annotations
    thresh = cm_normalized.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                f"{confusion_matrix[i, j]}\n({cm_normalized[i, j]:.1%})",
                ha="center",
                va="center",
                color="white" if cm_normalized[i, j] > thresh else "black",
                fontsize=9,
            )

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Matrice de confusion sauvegardée: {output_path}")


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    class_names: Optional[List[str]] = None,
    target_names: Optional[List[str]] = None,
) -> None:
    """Sauvegarde le rapport de classification en fichier texte.

    Args:
        y_true: Labels réels (one-hot encoded ou indices)
        y_pred: Prédictions (one-hot encoded ou indices)
        output_path: Chemin de sortie pour le rapport
        class_names: Liste des noms de classes (optionnel)
        target_names: Noms des classes pour le rapport (optionnel)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convertir en indices si nécessaire
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_indices = np.argmax(y_true, axis=1)
    else:
        y_true_indices = y_true.flatten().astype(int)

    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred_indices = np.argmax(y_pred, axis=1)
    else:
        y_pred_indices = y_pred.flatten().astype(int)

    # Utiliser target_names si fourni, sinon class_names
    report_target_names = target_names or class_names

    # Générer le rapport
    report = classification_report(
        y_true_indices,
        y_pred_indices,
        target_names=report_target_names,
        output_dict=False,
    )

    # Sauvegarder le rapport
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
        f.write("\n")

    logger.info(f"Rapport de classification sauvegardé: {output_path}")


def save_metrics(metrics: Dict, output_path: Path) -> None:
    """Sauvegarde les métriques en fichier JSON.

    Args:
        metrics: Dictionnaire contenant les métriques
        output_path: Chemin de sortie pour le fichier JSON
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    logger.info(f"Métriques sauvegardées: {output_path}")


def evaluate_model(
    model,
    test_generator,
    output_dir: Path,
    class_names: Optional[List[str]] = None,
    model_name: str = "baseline_model",
) -> Dict:
    """Évalue un modèle sur un générateur de test et sauvegarde toutes les métriques.

    Args:
        model: Modèle Keras compilé
        test_generator: Générateur de données de test
        output_dir: Répertoire de sortie pour les métriques
        class_names: Liste des noms de classes (optionnel)
        model_name: Nom du modèle pour les fichiers de sortie

    Returns:
        Dictionnaire contenant toutes les métriques calculées
    """
    logger.info("=" * 60)
    logger.info("Démarrage de l'évaluation du modèle")
    logger.info("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Obtenir les noms de classes depuis le générateur si disponibles
    if class_names is None and hasattr(test_generator, "get_class_names"):
        class_names = test_generator.get_class_names()

    # Collecter toutes les prédictions et labels
    logger.info("Collecte des prédictions sur le jeu de test...")
    y_true_list = []
    y_pred_list = []

    for batch_idx in range(len(test_generator)):
        X_batch, y_batch = test_generator[batch_idx]
        y_pred_batch = model.predict(X_batch, verbose=0)

        y_true_list.append(y_batch)
        y_pred_list.append(y_pred_batch)

    # Concaténer tous les batches
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)

    logger.info(f"Nombre d'échantillons évalués: {len(y_true)}")

    # Calculer les métriques
    logger.info("\nCalcul des métriques...")
    metrics = calculate_metrics(y_true, y_pred, class_names=class_names)

    # Sauvegarder les métriques
    metrics_path = output_dir / f"{model_name}_metrics.json"
    save_metrics(metrics, metrics_path)

    # Sauvegarder la matrice de confusion
    cm = np.array(metrics["confusion_matrix"])
    cm_path = output_dir / f"{model_name}_confusion_matrix.png"
    save_confusion_matrix(cm, cm_path, class_names=class_names)

    # Sauvegarder le rapport de classification
    report_path = output_dir / f"{model_name}_classification_report.txt"
    save_classification_report(y_true, y_pred, report_path, class_names=class_names)

    logger.info("\n" + "=" * 60)
    logger.info("Évaluation terminée!")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  F1-score (macro): {metrics['f1_macro']:.4f}")
    logger.info("=" * 60)

    return metrics

