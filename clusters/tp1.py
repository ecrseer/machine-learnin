import pandas as pd
import requests
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import mglearn
from sklearn.feature_extraction import text


def baixar_csv_github(url):
    print(f"Baixando arquivo de {url}...")
    if 'github.com' in url and '/blob/' in url:
        raw_url = url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
    else:
        raw_url = url
    response = requests.get(raw_url)
    if response.status_code == 200:
        return pd.read_csv(StringIO(response.text))
    else:
        raise Exception(f"Erro ao baixar arquivo: {response.status_code}")


url_sonar = "https://github.com/ecrseer/machine-learnin/blob/main/clusters/shopmania.csv"
df = baixar_csv_github(url_sonar)


df = df.sample(n=18420, random_state=42).reset_index(drop=True)

def vetorizar(vetorizador, atributo):
    atributo_vec = vetorizador.fit_transform(atributo)
    atributo_vec_df = pd.DataFrame(
        atributo_vec.toarray(), 
        columns=vetorizador.get_feature_names_out()
    )
    return atributo_vec_df

print("Arquivo baixado! iniciano vetorizando textos em numero...")
df.columns = ["id", "product_name", "some_number", "category"]
atributos_tfidf = vetorizar(text.TfidfVectorizer(stop_words="english"), df['product_name'])

print("vetorizado! inicnando o KMeans...")
kmeans = KMeans(n_clusters=3, random_state=0)


X = atributos_tfidf.values  

kmeans.fit(X)
y_pred = kmeans.predict(X)

print("Plotando clusters...")

plt.figure(figsize=(8,6))
plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y_pred,               
    cmap=mglearn.cm2,       
    s=60,
    edgecolor='k'
)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    c="red",
    marker="^",
    s=200,
    edgecolor="k",
    label="Centroids"
)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.title("KMeans Clustering with plt.scatter")
plt.legend()
plt.show()
