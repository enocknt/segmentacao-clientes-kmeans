# Copyright (c) 2026 The Project Authors.
# Licensed under the MIT License. See the LICENSE file for details.

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Constantes e configurações
RANDOM_SEED = 42
MAX_K = 10
K_CHOSEN = 5
NAVY, RED = "#1f3b73", "#c0392b"
PALETTE = ["#1f3b73", "#c0392b", "#27ae60", "#e67e22", "#8e44ad"]


def setup_directories() -> Tuple[Path, Path]:
    """
    Resolve os caminhos relativos ao script atual e certifica-se de que a 
    pasta 'figuras' exista na raiz do projeto.
    
    Returns:
        Tuple[Path, Path]: Caminho da pasta atual (src) e da pasta de figuras.
    """
    curr_dir = Path(__file__).parent
    fig_dir = curr_dir.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return curr_dir, fig_dir


def load_and_preprocess_data(curr_dir: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Carega os dados do shopping e aplica padronização Z-score para
    nivelar escalas das features (Age, Income, Spending).
    
    Args:
        curr_dir (Path): Pasta contendo o dataset 'Mall_Customers.csv'.
    """
    data_path = curr_dir / "Mall_Customers.csv"
    if not data_path.exists():
        logging.error(f"Erro Crítico: Arquivo dataset não encontrado em {data_path}.")
        sys.exit(1)
        
    df = pd.read_csv(data_path)
    df.columns = ["CustomerID", "Genre", "Age", "Income", "Spending"]
    X = df[["Age", "Income", "Spending"]].values
    Xs = StandardScaler().fit_transform(X)
    return df, X, Xs


def save_metric_plot(ks: list, values: list, vline_k: int, vline_val: float, 
                     label_text: str, xlabel: str, ylabel: str, filename: str, 
                     fig_dir: Path, arrow_xytext: Tuple[float, float]) -> None:
    """
    Plota e salva gráficos de linha customizados para análise de critérios (K-Means).
    """
    fig, a = plt.subplots(figsize=(3.4, 2.6))
    
    # Adaptar estilo de linha dependendo da métrica
    if "Inércia" in ylabel:
        fmt = "o-"
    elif "silhueta" in ylabel:
        fmt = "s-"
    else:
        fmt = "^-"
        
    a.plot(ks, values, fmt, color=NAVY, lw=1.8, ms=6)
    a.axvline(vline_k, ls="--", color=RED, lw=1.3)
    a.annotate(
        label_text, 
        xy=(vline_k, vline_val), 
        xytext=arrow_xytext,
        fontsize=9.5, 
        color=RED, 
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.1)
    )
    a.set_xlabel(xlabel)
    a.set_ylabel(ylabel)
    a.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / filename, bbox_inches="tight")
    plt.close()


def evaluate_metrics(Xs: np.ndarray, fig_dir: Path, curr_dir: Path, 
                     k_chosen: int, max_k: int, random_seed: int) -> None:
    """
    Treina o modelo K-Means para um range de K e computa métricas de inércia, 
    silhueta e índice Calinski-Harabasz. Gera os DataFrames e as Plots relativas.
    """
    ks = list(range(2, max_k + 1))
    inertia, sil, ch = [], [], []
    
    for k in ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=random_seed).fit(Xs)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(Xs, km.labels_))
        ch.append(calinski_harabasz_score(Xs, km.labels_))

    crit = pd.DataFrame({"k": ks, "Inertia": np.round(inertia, 2),
                         "Silhouette": np.round(sil, 4), "CH": np.round(ch, 2)})
    crit.to_csv(curr_dir / "criteria.csv", index=False)
    logging.info("\n" + crit.to_string(index=False))

    # (a) Cotovelo
    idx_5 = ks.index(k_chosen) if k_chosen in ks else 0
    save_metric_plot(
        ks, inertia, k_chosen, inertia[idx_5], 
        f"cotovelo\n$k={k_chosen}$", "Número de grupos $k$", "Inércia (WCSS)", 
        "fig_elbow.png", fig_dir, (6.2, inertia[3] + 60)
    )

    # (b) Silhueta
    bi = int(np.argmax(sil))
    best_sil = ks[bi]
    save_metric_plot(
        ks, sil, best_sil, sil[bi], 
        f"máximo\n$k={best_sil}$", "Número de grupos $k$", "Coeficiente de silhueta", 
        "fig_silhouette.png", fig_dir, (best_sil + 0.6, sil[bi] - 0.025)
    )

    # (c) Calinski-Harabasz
    bc = int(np.argmax(ch))
    best_ch = ks[bc]
    save_metric_plot(
        ks, ch, best_ch, ch[bc], 
        f"máximo\n$k={best_ch}$", "Número de grupos $k$", "Índice Calinski-Harabasz", 
        "fig_ch.png", fig_dir, (best_ch + 0.5, ch[bc] - 12)
    )

    logging.info(f"Melhor silhueta: {best_sil} | Melhor CH: {best_ch}")


def run_kmeans_and_pca(df: pd.DataFrame, Xs: np.ndarray, fig_dir: Path, 
                       curr_dir: Path, k_chosen: int, random_seed: int) -> None:
    """
    Treina o modelo K-Means final com K otimizado. Aplica PCA para 
    redução final de dimensionalidade, plota as personas e salva perfis (clusters).
    """
    km = KMeans(n_clusters=k_chosen, n_init=50, random_state=random_seed).fit(Xs)
    df["Cluster"] = km.labels_

    pca = PCA(n_components=2).fit(Xs)
    P = pca.transform(Xs)
    var = pca.explained_variance_ratio_
    logging.info(f"Variância PCA: {np.round(var, 3)} soma: {round(var.sum(), 3)}")

    fig, axp = plt.subplots(figsize=(4.6, 3.4))
    for c in range(k_chosen):
        m = df["Cluster"] == c
        axp.scatter(
            P[m, 0], P[m, 1], 
            s=22, color=PALETTE[c], 
            label=f"Cluster {c}", alpha=0.8, edgecolor="white", lw=0.3
        )
        
    cent = pca.transform(km.cluster_centers_)
    axp.scatter(cent[:, 0], cent[:, 1], marker="X", s=120, color="black", label="Centroides", zorder=5)
    axp.set_xlabel(f"PC1 ({var[0] * 100:.1f}%)")
    axp.set_ylabel(f"PC2 ({var[1] * 100:.1f}%)")
    axp.legend(fontsize=7, loc="best")
    axp.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_pca.png", bbox_inches="tight")
    plt.close()

    prof = df.groupby("Cluster")[["Age", "Income", "Spending"]].mean().round(1)
    prof["N"] = df.groupby("Cluster").size()
    prof.to_csv(curr_dir / "profiles.csv")
    logging.info("\n" + prof.to_string())

    score = silhouette_score(Xs, km.labels_)
    logging.info(f"Silhueta final k={k_chosen}: {round(score, 4)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de Segmentação K-Means.")
    parser.add_argument("--k_chosen", type=int, default=K_CHOSEN, help="Número ótimo de K escolhido")
    parser.add_argument("--max_k", type=int, default=MAX_K, help="Número máximo de K na exploração")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Semente aleatória")
    args = parser.parse_args()

    np.random.seed(args.seed)
    plt.rcParams.update({"font.size": 9, "figure.dpi": 200})
    
    curr_dir, fig_dir = setup_directories()
    df, X, Xs = load_and_preprocess_data(curr_dir)
    evaluate_metrics(Xs, fig_dir, curr_dir, args.k_chosen, args.max_k, args.seed)
    run_kmeans_and_pca(df, Xs, fig_dir, curr_dir, args.k_chosen, args.seed)


if __name__ == "__main__":
    main()
