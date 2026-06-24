# Copyright (c) 2026 The Project Authors.
# Licensed under the MIT License. See the LICENSE file for details.

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    """
    Função principal que gera o diagrama da arquitetura (fig_arch.png).
    O diagrama ilustra os 6 estágios do pipeline de segmentação.
    Garante a criação prévia do diretório '../figuras/' se não existir.
    """
    curr_dir = Path(__file__).parent
    fig_dir = curr_dir.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": 8.5})
    fig, ax = plt.subplots(figsize=(7.0, 1.7))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 3)
    ax.axis("off")
    
    stages = [
        "Dados do\ncliente\n(idade, renda,\nscore)",
        "Pré-proc.\n(padronização\nz-score)",
        "Seleção de k\n(cotovelo +\nsilhueta + CH)",
        "K-Means\n(agrupamento)",
        "Geração de\npersonas\n(rótulo +\nmarketing)",
        "Saída\nacionável"
    ]
    colors = ["#d6e0f0", "#d6e0f0", "#f6dcd0", "#f6dcd0", "#d4ecdc", "#d4ecdc"]
    x = 0.3
    w = 2.15
    gap = 0.25
    xs = []
    
    for s, c in zip(stages, colors):
        box = FancyBboxPatch(
            (x, 0.7), w, 1.6,
            boxstyle="round,pad=0.05,rounding_size=0.12",
            fc=c, ec="#34495e", lw=1.2
        )
        ax.add_patch(box)
        ax.text(x + w / 2, 1.5, s, ha="center", va="center", fontsize=7.5)
        xs.append(x)
        x += w + gap
        
    for i in range(len(stages) - 1):
        a = FancyArrowPatch(
            (xs[i] + w, 1.5), (xs[i + 1], 1.5),
            arrowstyle="-|>", mutation_scale=12,
            color="#34495e", lw=1.3
        )
        ax.add_patch(a)
        
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_arch.png", bbox_inches="tight", dpi=200)
    plt.close()
    
    logging.info("Diagrama de arquitetura gerado com sucesso.")


if __name__ == "__main__":
    main()
