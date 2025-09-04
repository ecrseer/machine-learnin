import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll



def cria_dataframe_exemplo():
    np.random.seed(123)
    variables = ['X', 'Y', 'Z']
    labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
    X = np.random.random_sample([5, 3]) * 10
    df = pd.DataFrame(X, columns=variables, index=labels)
    print(df)
    return df, labels

def cria_dataframe_swiss(n_samples=1000, noise=0.0, random_state=None, hole=False):
    X, t = make_swiss_roll(n_samples=n_samples, noise=noise,
                           random_state=random_state, hole=hole)
    df_swiss = pd.DataFrame(X, columns=['X', 'Y', 'Z'])
    print(f"Swiss Roll dataset gerado: {n_samples} amostras, noise={noise}, hole={hole}")
    print(df_swiss.head())
    return df_swiss, t


def calcula_clusters(df):
    row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')
    row_clusters = pd.DataFrame(
        row_clusters,
        columns=['row label 1', 'row label 2', 'distancia', 'itens no cluster'],
        index=['cluster %d' % (i + 1) for i in range(row_clusters.shape[0])]
    )
    print(row_clusters)
    return row_clusters


def plota_dendrograma(row_clusters, labels):
    row_dendr = dendrogram(row_clusters, labels=labels)
    plt.tight_layout()
    plt.ylabel('Euclidean distance')
    plt.show()


def plota_heatmap_clusters(df, row_clusters, labels):
    fig = plt.figure(figsize=(8, 8), facecolor='white')
    axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
    row_dendr = dendrogram(row_clusters, orientation='left')
    df_rowclust = df.iloc[row_dendr['leaves'][::-1]]
    axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
    cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')
    axd.set_xticks([])
    axd.set_yticks([])
    for i in axm.spines.values():
        i.set_visible(False)
    fig.colorbar(cax)
    axm.set_xticklabels([''] + list(df_rowclust.columns))
    axm.set_yticklabels([''] + list(df_rowclust.index))
    plt.show()
    return df_rowclust

labels = df.index.astype(str)

df, t = cria_dataframe_swiss(n_samples=1000, noise=0.05, random_state=42)
row_clusters = calcula_clusters(df)
plota_dendrograma(row_clusters, labels)
df_rowclust = plota_heatmap_clusters(df, row_clusters,labels)