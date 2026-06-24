# Segmentação de Clientes de Shopping com K-Means

Sistema de segmentação de clientes baseado em **K-Means**, com seleção criteriosa do número de grupos (cotovelo + silhueta + Calinski-Harabasz) e geração de **personas acionáveis** com recomendações de marketing.

Trabalho da Unidade III da disciplina **Aprendizado de Máquina Não-Supervisionado (IMD0029)** — Instituto Metrópole Digital (IMD), UFRN. Profa. Ligia Borges.

## Visão geral

O sistema é estruturado como um *pipeline*:

```
dados do cliente → pré-processamento (z-score) → seleção de k → K-Means → geração de personas → saída acionável
```

- **Entrada:** idade, renda anual e *spending score* (Mall Customer Segmentation Dataset, 200 clientes).
- **Núcleo:** K-Means (scikit-learn, k-means++, 50 inicializações, semente fixa).
- **Seleção de k:** combinação de três critérios — método do cotovelo, coeficiente de silhueta e índice Calinski-Harabasz.
- **Saída:** cada cluster é traduzido em uma persona com recomendação de marketing.

## Estrutura do repositório

```
.
├── artigo.tex            # Artigo técnico em LaTeX (formato IEEE)
├── figuras/              # Figuras geradas pelos scripts
│   ├── fig_arch.png      # Diagrama de arquitetura do pipeline
│   ├── fig_kmeans.png    # Funcionamento do algoritmo K-Means (4 etapas)
│   ├── fig_elbow.png     # Critério do cotovelo
│   ├── fig_silhouette.png# Coeficiente de silhueta
│   ├── fig_ch.png        # Índice Calinski-Harabasz
│   └── fig_pca.png       # Clusters projetados via PCA
├── src/
│   ├── pipeline.py       # Pipeline completo (pré-proc, k, K-Means, critérios, PCA, personas)
│   ├── arch.py           # Gera o diagrama de arquitetura
│   ├── gen_kmeans.py     # Gera a figura do funcionamento do K-Means
│   ├── Mall_Customers.csv# Dataset
│   ├── criteria.csv      # Resultados dos critérios (gerado)
│   └── profiles.csv      # Perfis dos clusters (gerado)
├── requirements.txt
├── .gitignore
└── README.md
```

## Como reproduzir os experimentos

```bash
# (opcional) ambiente virtual
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt

cd src
python pipeline.py    # gera fig_elbow, fig_silhouette, fig_ch, fig_pca e os CSVs
python gen_kmeans.py  # gera a figura do funcionamento do K-Means
python arch.py        # gera o diagrama de arquitetura
```

## Como compilar o artigo

O artigo usa a classe `IEEEtran` e o pacote de português do babel, ambos presentes em qualquer
distribuição TeX Live completa e no Overleaf (não é preciso instalar nada manualmente):

```bash
pdflatex artigo.tex
pdflatex artigo.tex   # segunda passada para resolver as referências
```

## Referências-base

- MacQueen (1967) — K-Means
- Rousseeuw (1987) — coeficiente de silhueta
- Caliński & Harabasz (1974) — índice CH
- Jolliffe (2002) — PCA
- Pedregosa et al. (2011) — scikit-learn

## Declaração de uso de IA

O uso de ferramentas de IA está descrito no anexo do artigo, conforme exigido pela disciplina.
