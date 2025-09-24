import pandas as pd
import requests
from io import StringIO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import seaborn as sns
import numpy as np

# ========================
# CONFIGURAÇÕES GLOBAIS
# ========================
FEATURES_3D = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]

# ========================
# FUNÇÃO PRINCIPAL
# ========================
def executar_analise_completa(url, k_clusters=4):
    """Executa análise completa de clustering."""
    print("Iniciando análise completa de clustering...")
    
    # 1. Carregar e preparar dados
    df_original, df, X, X_scaled, scaler = carregar_e_preparar_dados(url)
    
    # 2. Análise do método do cotovelo
    plotar_elbow_method(X_scaled)
    
    # 3. Treinar KMeans
    kmeans_model, labels = treinar_kmeans(X_scaled, k_clusters)
    
    # 4. Aplicar quantização vetorial
    aplicar_quantizacao_vetorial(X, k_clusters)
    
    # 5. Calcular distâncias
    df = calcular_distancias_centroides(df, kmeans_model, X_scaled)
    
    # 6. Visualizações KMeans
    plotar_clusters_3d(X, labels, kmeans_model, scaler)
    plotar_vistas_2d(X, labels, kmeans_model, scaler)
    
    # 7. Análise estatística KMeans
    analisar_clusters_estatisticas(df, labels)
    
    # 8. AgglomerativeClustering com diferentes linkages
    analisar_agglomerative_clustering(X_scaled, k_clusters)
    
    print("\nAnálise completa finalizada!")
    return df

def carregar_e_preparar_dados(url):
    """Carrega dados do GitHub e prepara para clustering."""
    print("Carregando dados...")
    df_original = baixar_csv_github(url)
    print("shape:", df_original.shape)
    print(df_original.columns)
    print(df_original.head())
    
    # Preparar dados
    if "CustomerID" in df_original.columns:
        df = df_original.drop(columns=["CustomerID"]).copy()
    else:
        df = df_original.copy()
    
    # Garantir nomes limpos
    df.columns = [c.strip() for c in df.columns]
    
    # Verificar features necessárias
    for f in FEATURES_3D:
        if f not in df.columns:
            raise RuntimeError(f"Coluna esperada '{f}' não encontrada no CSV.")
    
    # Remover linhas com missing
    df = df.dropna(subset=FEATURES_3D).reset_index(drop=True)
    
    # Criar matrizes X
    X = df[FEATURES_3D].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Dados preparados: {X.shape[0]} amostras, {X.shape[1]} features")
    return df_original, df, X, X_scaled, scaler

def baixar_csv_github(url):
    """Baixa CSV do GitHub (aceita blob URLs) e retorna DataFrame."""
    if 'github.com' in url and '/blob/' in url:
        raw_url = url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
    else:
        raw_url = url
    resp = requests.get(raw_url)
    resp.raise_for_status()
    return pd.read_csv(StringIO(resp.text))

def plotar_elbow_method(X_scaled, ks=range(1, 9)):
    """Plota gráfico do método do cotovelo para escolha de k."""
    print("Calculando método do cotovelo...")
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X_scaled)
        inertias.append(km.inertia_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(list(ks), inertias, "-o", linewidth=2, markersize=8)
    plt.xlabel("Número de Clusters (k)")
    plt.ylabel("Inércia")
    plt.title("Método do Cotovelo - Escolha do k ideal")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return inertias

def treinar_kmeans(X_scaled, k_clusters):
    """Treina modelo KMeans com k especificado."""
    print(f"Treinando KMeans com k={k_clusters}...")
    kmeans_model = KMeans(n_clusters=k_clusters, random_state=0, n_init=10)
    labels = kmeans_model.fit_predict(X_scaled)
    
    # Calcular silhouette score
    try:
        sil_score = silhouette_score(X_scaled, labels)
        print(f"KMeans k={k_clusters} — silhouette score: {sil_score:.6f}")
    except Exception:
        print("Erro ao calcular silhouette score")
    
    return kmeans_model, labels

def aplicar_quantizacao_vetorial(X, k_clusters):
    """Aplica quantização vetorial usando os centroides do KMeans."""
    print("Aplicando quantização vetorial...")
    
    # Treinar KMeans nos dados originais (não escalados) para quantização
    kmeans_quantizacao = KMeans(
        n_clusters=k_clusters, 
        random_state=0, 
        n_init=10
    )
    kmeans_quantizacao.fit(X)
    
    # Quantizar: substituir cada ponto pelo centroide mais próximo
    X_quantizado = kmeans_quantizacao.cluster_centers_[kmeans_quantizacao.predict(X)]
    
    # Calcular erro de quantização (distorção)
    erro_quantizacao = np.mean(np.sum((X - X_quantizado)**2, axis=1))
    print(f"Erro médio de quantização: {erro_quantizacao:.4f}")
    
    # Visualizar quantização (apenas 2D para simplicidade)
    plotar_quantizacao_2d(X, X_quantizado, kmeans_quantizacao.cluster_centers_)
    
    return X_quantizado

def plotar_quantizacao_2d(X, X_quantizado, centroides):
    """Plota visualização 2D da quantização vetorial."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Age vs Annual Income
    axes[0].scatter(X[:, 0], X[:, 1], c='lightblue', 
                   alpha=0.6, label='Dados Originais', s=50)
    axes[0].scatter(X_quantizado[:, 0], X_quantizado[:, 1], 
                   c='red', alpha=0.8, label='Dados Quantizados', s=30)
    axes[0].scatter(centroides[:, 0], centroides[:, 1],
                   c='black', marker='^', s=200, 
                   label='Centroides', edgecolor='white', linewidth=2)
    axes[0].set_xlabel(FEATURES_3D[0])
    axes[0].set_ylabel(FEATURES_3D[1])
    axes[0].set_title("Quantização: Age vs Annual Income")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Annual Income vs Spending Score
    axes[1].scatter(X[:, 1], X[:, 2], c='lightblue', 
                   alpha=0.6, label='Dados Originais', s=50)
    axes[1].scatter(X_quantizado[:, 1], X_quantizado[:, 2], 
                   c='red', alpha=0.8, label='Dados Quantizados', s=30)
    axes[1].scatter(centroides[:, 1], centroides[:, 2],
                   c='black', marker='^', s=200, 
                   label='Centroides', edgecolor='white', linewidth=2)
    axes[1].set_xlabel(FEATURES_3D[1])
    axes[1].set_ylabel(FEATURES_3D[2])
    axes[1].set_title("Quantização: Income vs Spending Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def calcular_distancias_centroides(df, kmeans_model, X_scaled):
    """Calcula distâncias dos pontos aos centroides mais próximos."""
    distances = kmeans_model.transform(X_scaled)
    dist_min = distances.min(axis=1)
    df["distancia_cluster"] = dist_min
    
    print("\nExemplo de distâncias calculadas:")
    print(df[FEATURES_3D + ["distancia_cluster"]].head())
    
    return df

def plotar_clusters_3d(X, labels, kmeans_model, scaler):
    """Plota visualização 3D dos clusters."""
    # Converter centroides de volta ao espaço original
    centers_orig = scaler.inverse_transform(kmeans_model.cluster_centers_)
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot dos pontos coloridos por cluster
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], 
                       c=labels, cmap=plt.cm.tab10, 
                       s=60, edgecolor='k', alpha=0.7)
    
    # Plot dos centroides
    ax.scatter(centers_orig[:, 0], centers_orig[:, 1], centers_orig[:, 2], 
              c="red", marker="^", s=200, edgecolor="k", label="Centroides")
    
    # Labels dos eixos
    ax.set_xlabel(FEATURES_3D[0])
    ax.set_ylabel(FEATURES_3D[1])
    ax.set_zlabel(FEATURES_3D[2])
    ax.set_title(f"KMeans (k={kmeans_model.n_clusters}) - Visualização 3D")
    
    ax.legend()
    plt.tight_layout()
    plt.show()

def plotar_vistas_2d(X, labels, kmeans_model, scaler):
    """Plota múltiplas vistas 2D dos clusters."""
    centers_orig = scaler.inverse_transform(kmeans_model.cluster_centers_)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Vista XY (Age vs Annual Income)
    axes[0].scatter(X[:, 0], X[:, 1], c=labels, 
                   cmap=plt.cm.tab10, s=60, edgecolor='k', alpha=0.7)
    axes[0].scatter(centers_orig[:, 0], centers_orig[:, 1], 
                   c="red", marker="^", s=200, edgecolor="k")
    axes[0].set_xlabel(FEATURES_3D[0])
    axes[0].set_ylabel(FEATURES_3D[1])
    axes[0].set_title(f"Vista XY: {FEATURES_3D[0]} vs {FEATURES_3D[1]}")
    axes[0].grid(True, alpha=0.3)
    
    # Vista XZ (Age vs Spending Score)
    axes[1].scatter(X[:, 0], X[:, 2], c=labels, 
                   cmap=plt.cm.tab10, s=60, edgecolor='k', alpha=0.7)
    axes[1].scatter(centers_orig[:, 0], centers_orig[:, 2], 
                   c="red", marker="^", s=200, edgecolor="k")
    axes[1].set_xlabel(FEATURES_3D[0])
    axes[1].set_ylabel(FEATURES_3D[2])
    axes[1].set_title(f"Vista XZ: {FEATURES_3D[0]} vs {FEATURES_3D[2]}")
    axes[1].grid(True, alpha=0.3)
    
    # Vista YZ (Annual Income vs Spending Score)
    axes[2].scatter(X[:, 1], X[:, 2], c=labels, 
                   cmap=plt.cm.tab10, s=60, edgecolor='k', alpha=0.7)
    axes[2].scatter(centers_orig[:, 1], centers_orig[:, 2], 
                   c="red", marker="^", s=200, edgecolor="k")
    axes[2].set_xlabel(FEATURES_3D[1])
    axes[2].set_ylabel(FEATURES_3D[2])
    axes[2].set_title(f"Vista YZ: {FEATURES_3D[1]} vs {FEATURES_3D[2]}")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analisar_clusters_estatisticas(df, labels):
    """Calcula e exibe estatísticas descritivas de cada cluster."""
    df_temp = df.copy()
    df_temp['cluster'] = labels
    
    print("\n" + "="*60)
    print("ANÁLISE ESTATÍSTICA DOS CLUSTERS - KMEANS")
    print("="*60)
    
    for cluster_id in sorted(df_temp['cluster'].unique()):
        cluster_data = df_temp[df_temp['cluster'] == cluster_id]
        print(f"\nCluster {cluster_id} ({len(cluster_data)} pontos):")
        
        for feature in FEATURES_3D:
            media = cluster_data[feature].mean()
            desvio = cluster_data[feature].std()
            print(f"  {feature}: {media:.1f} ± {desvio:.1f}")
        
        if "distancia_cluster" in cluster_data.columns:
            dist_media = cluster_data["distancia_cluster"].mean()
            print(f"  Distância média ao centroide: {dist_media:.3f}")

def analisar_agglomerative_clustering(X_scaled, n_clusters=4):
    """Aplica AgglomerativeClustering com diferentes tipos de linkage."""
    print("\n" + "="*80)
    print("ANÁLISE COM AGGLOMERATIVE CLUSTERING")
    print("="*80)
    
    # Diferentes tipos de linkage para testar
    linkage_types = ['ward', 'average', 'single', 'complete']
    resultados_agg = {}
    
    # Aplicar cada tipo de linkage
    for linkage_type in linkage_types:
        print(f"\nTestando linkage: {linkage_type}")
        
        # Treinar modelo
        agg_model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_type
        )
        labels_agg = agg_model.fit_predict(X_scaled)
        
        # Calcular silhouette score
        try:
            sil_score = silhouette_score(X_scaled, labels_agg)
            print(f"Silhouette Score ({linkage_type}): {sil_score:.6f}")
        except Exception:
            sil_score = None
            print(f"Erro ao calcular silhouette score para {linkage_type}")
        
        # Armazenar resultados
        resultados_agg[linkage_type] = {
            'model': agg_model,
            'labels': labels_agg,
            'silhouette': sil_score
        }
    
    # Criar dendrogramas
    plotar_dendrogramas(X_scaled, linkage_types)
    
    # Plotar comparação dos clusters
    plotar_comparacao_agglomerative(X_scaled, resultados_agg)
    
    # Criar mapa de calor
    criar_mapa_calor_clusters(X_scaled, resultados_agg)
    
    return resultados_agg

def plotar_dendrogramas(X_scaled, linkage_types):
    """Plota dendrogramas para diferentes tipos de linkage."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, linkage_type in enumerate(linkage_types):
        print(f"Criando dendrograma para linkage: {linkage_type}")
        
        # Calcular linkage matrix
        if linkage_type == 'ward':
            # Ward precisa de distância euclidiana
            linkage_matrix = linkage(X_scaled, method=linkage_type)
        else:
            # Outros métodos podem usar diferentes distâncias
            linkage_matrix = linkage(X_scaled, method=linkage_type)
        
        # Plotar dendrograma
        dendrogram(
            linkage_matrix,
            ax=axes[i],
            truncate_mode='level',
            p=6,  # Mostrar apenas 6 níveis para clareza
            show_leaf_counts=True,
            leaf_font_size=8
        )
        
        axes[i].set_title(f'Dendrograma - Linkage: {linkage_type.title()}', 
                         fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Índice da Amostra ou (Tamanho do Cluster)')
        axes[i].set_ylabel('Distância')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plotar_comparacao_agglomerative(X_scaled, resultados_agg):
    """Plota comparação visual dos diferentes tipos de linkage."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (linkage_type, resultado) in enumerate(resultados_agg.items()):
        labels = resultado['labels']
        sil_score = resultado['silhouette']
        
        # Plot usando as duas primeiras dimensões para visualização
        scatter = axes[i].scatter(
            X_scaled[:, 0], X_scaled[:, 1], 
            c=labels, cmap=plt.cm.tab10, 
            s=50, alpha=0.7, edgecolor='k', linewidth=0.5
        )
        
        title = f'Agglomerative - {linkage_type.title()}'
        if sil_score is not None:
            title += f'\nSilhouette: {sil_score:.3f}'
        
        axes[i].set_title(title, fontsize=11, fontweight='bold')
        axes[i].set_xlabel(f'{FEATURES_3D[0]} (Normalizado)')
        axes[i].set_ylabel(f'{FEATURES_3D[1]} (Normalizado)')
        axes[i].grid(True, alpha=0.3)
        
        # Adicionar colorbar
        plt.colorbar(scatter, ax=axes[i], label='Cluster')
    
    plt.tight_layout()
    plt.show()

def criar_mapa_calor_clusters(X_scaled, resultados_agg):
    """Cria mapa de calor mostrando a distribuição dos clusters."""
    # Criar dataframe com os dados escalados
    df_heatmap = pd.DataFrame(X_scaled, columns=[f'{feature}_scaled' for feature in FEATURES_3D])
    
    # Adicionar labels de cada método
    for linkage_type, resultado in resultados_agg.items():
        df_heatmap[f'cluster_{linkage_type}'] = resultado['labels']
    
    # Calcular médias por cluster para cada método
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (linkage_type, resultado) in enumerate(resultados_agg.items()):
        # Preparar dados para heatmap
        cluster_means = []
        cluster_labels = []
        
        for cluster_id in sorted(resultado['labels']):
            mask = resultado['labels'] == cluster_id
            if np.any(mask):  # Verificar se cluster existe
                cluster_data = X_scaled[mask]
                cluster_mean = cluster_data.mean(axis=0)
                cluster_means.append(cluster_mean)
                cluster_labels.append(f'Cluster {cluster_id}')
        
        # Converter para array
        if cluster_means:
            heatmap_data = np.array(cluster_means)
            
            # Criar heatmap
            sns.heatmap(
                heatmap_data,
                xticklabels=[f'{feature}' for feature in FEATURES_3D],
                yticklabels=cluster_labels,
                annot=True,
                fmt='.2f',
                cmap='RdYlBu_r',
                ax=axes[i],
                cbar_kws={'label': 'Valor Médio (Normalizado)'}
            )
            
            sil_score = resultado['silhouette']
            title = f'Mapa de Calor - {linkage_type.title()}'
            if sil_score is not None:
                title += f' (Sil: {sil_score:.3f})'
            
            axes[i].set_title(title, fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Features')
            axes[i].set_ylabel('Clusters')
    
    plt.tight_layout()
    plt.show()
    
    # Criar também um heatmap comparativo dos silhouette scores
    criar_heatmap_silhouette_scores(resultados_agg)

def criar_heatmap_silhouette_scores(resultados_agg):
    """Cria heatmap comparando os silhouette scores dos diferentes métodos."""
    # Preparar dados para comparação
    scores_data = []
    methods = []
    
    for linkage_type, resultado in resultados_agg.items():
        if resultado['silhouette'] is not None:
            scores_data.append([resultado['silhouette']])
            methods.append(linkage_type.title())
    
    if scores_data:
        scores_matrix = np.array(scores_data)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            scores_matrix,
            xticklabels=['Silhouette Score'],
            yticklabels=methods,
            annot=True,
            fmt='.4f',
            cmap='viridis',
            cbar_kws={'label': 'Silhouette Score'}
        )
        
        plt.title('Comparação de Silhouette Scores\nAgglomerative Clustering', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Métrica')
        plt.ylabel('Método de Linkage')
        plt.tight_layout()
        plt.show()

# ========================
# EXECUÇÃO PRINCIPAL
# ========================
if __name__ == "__main__":
    # URL dos dados
    url = "https://raw.githubusercontent.com/ecrseer/machine-learnin/main/clusters/mall/Mall_Customers.csv"
    
    # Executar análise completa
    df_resultado = executar_analise_completa(url, k_clusters=4)