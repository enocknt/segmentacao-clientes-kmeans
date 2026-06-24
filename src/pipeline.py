import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA

np.random.seed(42)
plt.rcParams.update({"font.size": 9, "figure.dpi": 200})

df = pd.read_csv("Mall_Customers.csv")
df.columns = ["CustomerID", "Genre", "Age", "Income", "Spending"]
X = df[["Age", "Income", "Spending"]].values
Xs = StandardScaler().fit_transform(X)

# ---- Seleção de k: cotovelo, silhueta, Calinski-Harabasz ----
ks = list(range(2, 11))
inertia, sil, ch = [], [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(Xs)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(Xs, km.labels_))
    ch.append(calinski_harabasz_score(Xs, km.labels_))

crit = pd.DataFrame({"k": ks, "Inertia": np.round(inertia,2),
                     "Silhouette": np.round(sil,4), "CH": np.round(ch,2)})
crit.to_csv("criteria.csv", index=False)
print(crit.to_string(index=False))

# Três critérios em figuras SEPARADAS (cotovelo, silhueta, Calinski-Harabasz)
NAVY, RED = "#1f3b73", "#c0392b"

# (a) Cotovelo
fig, a = plt.subplots(figsize=(3.4, 2.6))
a.plot(ks, inertia, "o-", color=NAVY, lw=1.8, ms=6)
a.axvline(5, ls="--", color=RED, lw=1.3)
a.annotate("cotovelo\n$k=5$", xy=(5, inertia[3]), xytext=(6.2, inertia[3]+60),
           fontsize=9.5, color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
a.set_xlabel("Número de grupos $k$"); a.set_ylabel("Inércia (WCSS)"); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("../figuras/fig_elbow.png", bbox_inches="tight"); plt.close()

# (b) Silhueta
bi = int(np.argmax(sil))
fig, a = plt.subplots(figsize=(3.4, 2.6))
a.plot(ks, sil, "s-", color=NAVY, lw=1.8, ms=6)
a.axvline(ks[bi], ls="--", color=RED, lw=1.3)
a.annotate(f"máximo\n$k={ks[bi]}$", xy=(ks[bi], sil[bi]), xytext=(ks[bi]+0.6, sil[bi]-0.025),
           fontsize=9.5, color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
a.set_xlabel("Número de grupos $k$"); a.set_ylabel("Coeficiente de silhueta"); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("../figuras/fig_silhouette.png", bbox_inches="tight"); plt.close()

# (c) Calinski-Harabasz
bc = int(np.argmax(ch))
fig, a = plt.subplots(figsize=(3.4, 2.6))
a.plot(ks, ch, "^-", color=NAVY, lw=1.8, ms=7)
a.axvline(ks[bc], ls="--", color=RED, lw=1.3)
a.annotate(f"máximo\n$k={ks[bc]}$", xy=(ks[bc], ch[bc]), xytext=(ks[bc]+0.5, ch[bc]-12),
           fontsize=9.5, color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
a.set_xlabel("Número de grupos $k$"); a.set_ylabel("Índice Calinski-Harabasz"); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("../figuras/fig_ch.png", bbox_inches="tight"); plt.close()

best_sil = ks[int(np.argmax(sil))]
best_ch = ks[int(np.argmax(ch))]
print("Melhor silhueta:", best_sil, "| Melhor CH:", best_ch)

# ---- k escolhido = 5 (cotovelo claro; equilíbrio entre métricas e interpretabilidade) ----
K = 5
km = KMeans(n_clusters=K, n_init=50, random_state=42).fit(Xs)
df["Cluster"] = km.labels_

# ---- PCA p/ visualização 2D ----
pca = PCA(n_components=2).fit(Xs)
P = pca.transform(Xs)
var = pca.explained_variance_ratio_
print("Variância PCA:", np.round(var,3), "soma:", round(var.sum(),3))

palette = ["#1f3b73", "#c0392b", "#27ae60", "#e67e22", "#8e44ad"]
fig, axp = plt.subplots(figsize=(4.6, 3.4))
for c in range(K):
    m = df["Cluster"]==c
    axp.scatter(P[m,0], P[m,1], s=22, color=palette[c], label=f"Cluster {c}", alpha=.8, edgecolor="white", lw=.3)
cent = pca.transform(km.cluster_centers_)
axp.scatter(cent[:,0], cent[:,1], marker="X", s=120, color="black", label="Centroides", zorder=5)
axp.set_xlabel(f"PC1 ({var[0]*100:.1f}%)"); axp.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
axp.legend(fontsize=7, loc="best"); axp.grid(alpha=.3)
plt.tight_layout(); plt.savefig("../figuras/fig_pca.png", bbox_inches="tight"); plt.close()

# ---- Perfis dos clusters (médias em escala original) ----
prof = df.groupby("Cluster")[["Age","Income","Spending"]].mean().round(1)
prof["N"] = df.groupby("Cluster").size()
prof.to_csv("profiles.csv")
print(prof.to_string())

# Silhueta final
print("Silhueta k=5:", round(silhouette_score(Xs, km.labels_),4))
