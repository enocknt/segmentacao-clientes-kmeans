import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.spatial import Voronoi, voronoi_plot_2d

np.random.seed(7)
plt.rcParams.update({"font.size": 10, "font.family": "serif"})

# Dados sintéticos 2D para ilustrar o algoritmo
Xb, _ = make_blobs(n_samples=150, centers=3, cluster_std=1.1, random_state=7)

NAVY="#1f3b73"; RED="#c0392b"; GREEN="#27ae60"; GREY="#7f8c8d"
cols=[NAVY, RED, GREEN]

def assign(X, C):
    d = np.linalg.norm(X[:,None,:]-C[None,:,:], axis=2)
    return d.argmin(1)

# Estados do algoritmo
C0 = np.array([[-8.,8.],[-6.,6.],[-9.,5.]])  # inicialização ruim de propósito
lab0 = assign(Xb, C0)
C1 = np.array([Xb[lab0==j].mean(0) if (lab0==j).any() else C0[j] for j in range(3)])
lab1 = assign(Xb, C1)
C2 = np.array([Xb[lab1==j].mean(0) for j in range(3)])
lab2 = assign(Xb, C2)
# convergência
C = C2.copy()
for _ in range(10):
    lab = assign(Xb, C)
    Cn = np.array([Xb[lab==j].mean(0) for j in range(3)])
    if np.allclose(Cn, C): break
    C = Cn
labf = assign(Xb, C)

panels = [
    ("(a) Inicialização", C0, np.full(len(Xb), -1), C0),
    ("(b) Atribuição", C0, lab0, C0),
    ("(c) Atualização", C1, lab0, C0),
    ("(d) Convergência", C, labf, None),
]

fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.1))
for ax,(title,Cc,lab,Cold) in zip(axes, panels):
    if (lab<0).all():
        ax.scatter(Xb[:,0], Xb[:,1], s=14, color=GREY, alpha=.6, edgecolor="white", lw=.3)
    else:
        for j in range(3):
            m=lab==j
            ax.scatter(Xb[m,0], Xb[m,1], s=14, color=cols[j], alpha=.65, edgecolor="white", lw=.3)
    # centroides antigos (seta de movimento)
    if Cold is not None and title.startswith("(c)"):
        for j in range(3):
            ax.annotate("", xy=(Cc[j,0],Cc[j,1]), xytext=(Cold[j,0],Cold[j,1]),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.3))
    ax.scatter(Cc[:,0], Cc[:,1], marker="X", s=140, color="black",
               edgecolor="white", lw=1.2, zorder=5)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_edgecolor("#cccccc")
plt.tight_layout()
plt.savefig("../figuras/fig_kmeans.png", bbox_inches="tight", dpi=200); plt.close()
print("Diagrama K-Means OK")
