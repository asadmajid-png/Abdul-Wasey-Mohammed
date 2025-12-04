#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bank Marketing – Machine Learning Project

This script implements the pipeline described in the assignment:
- Loads the UCI Bank Marketing dataset
- Preprocesses data (target encoding, one-hot encoding, scaling)
- Trains and evaluates three classification models:
    * Logistic Regression
    * K-Nearest Neighbours (KNN)
    * Support Vector Machine (SVM, RBF kernel)
- Computes metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
- Plots ROC curves for the classifiers
- Performs K-Means clustering (k=4) + PCA for 2D visualisation

Usage (from terminal):
    python bank_marketing_ml.py --data bank-additional-full.csv

Make sure the CSV file is in the same folder or provide the full path.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ----------------------------------------------------------------------
# 1. DATA LOADING & BASIC EXPLORATION
# ----------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """
    Load the Bank Marketing dataset (semicolon-separated CSV).

    Parameters
    ----------
    path : str
        Path to 'bank-additional-full.csv'

    Returns
    -------
    df : pandas.DataFrame
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    print(f"[INFO] Loading dataset from: {path}")
    df = pd.read_csv(path, sep=';')
    print(f"[INFO] Dataset shape: {df.shape}")
    print("[INFO] First 5 rows:")
    print(df.head())
    print("\n[INFO] Columns:")
    print(df.columns.tolist())
    print("\n[INFO] Target value counts:")
    print(df['y'].value_counts())
    print("\n[INFO] Missing values per column:")
    print(df.isna().sum())
    print("-" * 70)
    return df


# ----------------------------------------------------------------------
# 2. PREPROCESSING
# ----------------------------------------------------------------------

def preprocess_data(df: pd.DataFrame):
    """
    Preprocess the Bank Marketing data:
    - Encode target y ('yes' -> 1, 'no' -> 0)
    - One-hot encode categorical features

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix after one-hot encoding
    y : pandas.Series
        Encoded target
    """
    print("[INFO] Preprocessing data...")

    # Encode target
    df = df.copy()
    df['y'] = df['y'].map({'no': 0, 'yes': 1})

    # Separate features and target
    X = df.drop('y', axis=1)
    y = df['y']

    # One-hot encode categorical variables
    X = pd.get_dummies(X, drop_first=True)
    print(f"[INFO] Shape of X after one-hot encoding: {X.shape}")
    print("-" * 70)
    return X, y


def split_data(X, y, test_size=0.3, random_state=42):
    """
    Train-test split with stratification.

    Parameters
    ----------
    X : pandas.DataFrame
    y : pandas.Series
    test_size : float
    random_state : int

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    print(f"[INFO] Splitting data (train={1-test_size:.0%}, test={test_size:.0%})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print("[INFO] Train size:", X_train.shape)
    print("[INFO] Test size :", X_test.shape)
    print("-" * 70)
    return X_train, X_test, y_train, y_test


# ----------------------------------------------------------------------
# 3. MODEL DEFINITIONS
# ----------------------------------------------------------------------

def build_models(random_state=42):
    """
    Create classification models wrapped in sklearn Pipelines
    with StandardScaler where needed.

    Returns
    -------
    models : dict
        Dictionary of model name -> Pipeline
    """
    print("[INFO] Building models...")

    models = {}

    # Logistic Regression
    models['Logistic Regression'] = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=random_state))
    ])

    # K-Nearest Neighbours
    models['KNN (k=5)'] = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=5))
    ])

    # Support Vector Machine (RBF kernel)
    models['SVM (RBF)'] = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', probability=True, random_state=random_state))
    ])

    print("[INFO] Models defined:", list(models.keys()))
    print("-" * 70)
    return models


# ----------------------------------------------------------------------
# 4. TRAINING & EVALUATION
# ----------------------------------------------------------------------

def evaluate_models(models, X_train, X_test, y_train, y_test):
    """
    Train and evaluate models. Prints metrics and returns a summary DataFrame.

    Parameters
    ----------
    models : dict
        Name -> sklearn Pipeline
    X_train, X_test, y_train, y_test

    Returns
    -------
    results_df : pandas.DataFrame
        Summary metrics for each model
    """
    print("[INFO] Training and evaluating models...")
    results = []

    for name, model in models.items():
        print("=" * 70)
        print(f"Model: {name}")
        print("=" * 70)

        # Fit
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Probabilities for ROC-AUC
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = None

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        if y_proba is not None:
            roc_auc = roc_auc_score(y_test, y_proba)
        else:
            roc_auc = np.nan

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-score": f1,
            "ROC-AUC": roc_auc
        })

        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-score : {f1:.4f}")
        if not np.isnan(roc_auc):
            print(f"ROC-AUC  : {roc_auc:.4f}")
        else:
            print("ROC-AUC  : N/A")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\n")

    results_df = pd.DataFrame(results)
    print("[INFO] Summary of model performance:")
    print(results_df)
    print("-" * 70)
    return results_df


def plot_roc_curves(models, X_test, y_test, output_path="roc_curves.png"):
    """
    Plot ROC curves for all models that support predict_proba and save as PNG.

    Parameters
    ----------
    models : dict
    X_test : pandas.DataFrame
    y_test : pandas.Series
    output_path : str
    """
    print(f"[INFO] Plotting ROC curves -> {output_path}")
    plt.figure(figsize=(8, 6))

    has_curve = False
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            RocCurveDisplay.from_predictions(y_test, y_proba, name=name)
            has_curve = True

    if not has_curve:
        print("[WARNING] None of the models support predict_proba. ROC curves not plotted.")
        return

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title("ROC Curves – Bank Marketing")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print("[INFO] ROC curves saved.")
    print("-" * 70)


# ----------------------------------------------------------------------
# 5. K-MEANS CLUSTERING + PCA VISUALISATION
# ----------------------------------------------------------------------

def perform_clustering(X, k=4, random_state=42,
                       cluster_plot_path="kmeans_pca_clusters.png"):
    """
    Perform K-Means clustering on scaled features and visualise using PCA (2D).

    Parameters
    ----------
    X : pandas.DataFrame
        Features after one-hot encoding
    k : int
        Number of clusters
    random_state : int
    cluster_plot_path : str
        Path to save the PCA scatter plot

    Returns
    -------
    cluster_labels : np.ndarray
        Cluster labels for each instance
    """
    print("[INFO] Performing K-Means clustering...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    print("[INFO] Cluster counts:")
    unique, counts = np.unique(clusters, return_counts=True)
    for c, cnt in zip(unique, counts):
        print(f"  Cluster {c}: {cnt} samples")

    # PCA for 2D visualisation
    print("[INFO] Running PCA (2 components) for visualisation...")
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, s=5)
    plt.title(f"K-Means Clusters (k={k}) with PCA (2D)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    legend1 = plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.gca().add_artist(legend1)
    plt.tight_layout()
    plt.savefig(cluster_plot_path, dpi=300)
    plt.close()

    print(f"[INFO] Cluster plot saved to {cluster_plot_path}")
    print("-" * 70)
    return clusters


# ----------------------------------------------------------------------
# 6. MAIN
# ----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Machine Learning Analysis of the Bank Marketing Dataset"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to bank-additional-full.csv"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.3,
        help="Test set proportion (default: 0.3)"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=4,
        help="Number of clusters for K-Means (default: 4)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load data
    df = load_data(args.data)

    # 2. Preprocess
    X, y = preprocess_data(df)

    # 3. Train-test split
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    # 4. Build models
    models = build_models(random_state=args.random_state)

    # 5. Train & evaluate
    results_df = evaluate_models(models, X_train, X_test, y_train, y_test)

    # Optionally save metrics to CSV for the report
    results_csv_path = "model_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"[INFO] Model results saved to {results_csv_path}")

    # 6. ROC curves
    plot_roc_curves(models, X_test, y_test, output_path="roc_curves.png")

    # 7. K-Means clustering + PCA
    _ = perform_clustering(
        X,
        k=args.clusters,
        random_state=args.random_state,
        cluster_plot_path="kmeans_pca_clusters.png"
    )

    print("[INFO] Pipeline completed successfully.")


if __name__ == "__main__":
    main()
