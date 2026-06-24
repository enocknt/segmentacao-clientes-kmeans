# Copyright (c) 2026 The Project Authors.
# Licensed under the MIT License. See the LICENSE file for details.

import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

logging.basicConfig(level=logging.INFO, format="%(message)s")


def assign(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Atribui os pontos do conjunto X aos centroides C mais próximos com base
    na distância Euclidiana.
    
    Args:
        X (np.ndarray): Posições das amostras (shape N x 2).
        C (np.ndarray): Posições atuais dos centroides (shape K x 2).
        
    Returns:
        np.ndarray: Um array 1D contendo o índice do centroide mais próximo 
        para cada amostra.
    """
    d = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
    return d.argmin(1)


def main() -> None:
    """
    Gera a figura ilustrativa mostrando o funcionamento detalhado do algoritmo 
    K-Means em dados sintéticos (fig_kmeans.png). Representa as fases de 
    inicialização, atribuição, atualização e convergência.
    """
    curr_dir = Path(__file__).parent
    fig_dir = curr_dir.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(7)
    plt.rcParams.update({"font.size": 10, "font.family": "serif"})

    # Dados sintéticos 2D para ilustrar o algoritmo
    Xb, _ = make_blobs(n_samples=150, centers=3, cluster_std=1.1, random_state=7)

    NAVY = "#1f3b73"
    RED = "#c0392b"
    GREEN = "#27ae60"
    GREY = "#7f8c8d"
    cols = [NAVY, RED, GREEN]

    # Estados do algoritmo
    C0 = np.array([[-8., 8.], [-6., 6.], [-9., 5.]])  # inicialização ruim de propósito
    lab0 = assign(Xb, C0)
    
    C1 = np.array([Xb[lab0 == j].mean(0) if (lab0 == j).any() else C0[j] for j in range(3)])
    lab1 = assign(Xb, C1)
    
    C2 = np.array([Xb[lab1 == j].mean(0) for j in range(3)])
    # Convergência
    C = C2.copy()
    for _ in range(10):
        lab = assign(Xb, C)
        Cn = np.array([Xb[lab == j].mean(0) for j in range(3)])
        if np.allclose(Cn, C):
            break
        C = Cn
    labf = assign(Xb, C)

    panels = [
        ("(a) Inicialização", C0, np.full(len(Xb), -1), C0),
        ("(b) Atribuição", C0, lab0, C0),
        ("(c) Atualização", C1, lab0, C0),
        ("(d) Convergência", C, labf, None),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.1))
    for ax, (title, Cc, lab, Cold) in zip(axes, panels):
        if (lab < 0).all():
            ax.scatter(Xb[:, 0], Xb[:, 1], s=14, color=GREY, alpha=0.6, edgecolor="white", lw=0.3)
        else:
            for j in range(3):
                m = lab == j
                ax.scatter(Xb[m, 0], Xb[m, 1], s=14, color=cols[j], alpha=0.65, edgecolor="white", lw=0.3)
                
        # Centroides antigos (seta de movimento)
        if Cold is not None and title.startswith("(c)"):
            for j in range(3):
                ax.annotate("", xy=(Cc[j, 0], Cc[j, 1]), xytext=(Cold[j, 0], Cold[j, 1]),
                            arrowprops=dict(arrowstyle="->", color="black", lw=1.3))
                
        ax.scatter(Cc[:, 0], Cc[:, 1], marker="X", s=140, color="black",
                   edgecolor="white", lw=1.2, zorder=5)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#cccccc")
            
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_kmeans.png", bbox_inches="tight", dpi=200)
    plt.close()
    
    logging.info("Diagrama K-Means OK")


if __name__ == "__main__":
    main()
