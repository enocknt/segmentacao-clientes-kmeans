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

# Figura 2: três critérios
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2))
ax[0].plot(ks, inertia, "o-", color="#1f3b73"); ax[0].set_title("(a) Cotovelo"); ax[0].set_xlabel("k"); ax[0].set_ylabel("Inércia (WCSS)")
ax[0].axvline(5, ls="--", color="#c0392b", lw=1)
ax[1].plot(ks, sil, "s-", color="#1f3b73"); ax[1].set_title("(b) Silhueta"); ax[1].set_xlabel("k"); ax[1].set_ylabel("Coef. silhueta")
ax[1].axvline(ks[int(np.argmax(sil))], ls="--", color="#c0392b", lw=1)
ax[2].plot(ks, ch, "^-", color="#1f3b73"); ax[2].set_title("(c) Calinski-Harabasz"); ax[2].set_xlabel("k"); ax[2].set_ylabel("Índice CH")
ax[2].axvline(ks[int(np.argmax(ch))], ls="--", color="#c0392b", lw=1)
for a in ax: a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("../figuras/fig_criteria.png", bbox_inches="tight"); plt.close()

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
