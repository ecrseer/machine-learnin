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

 
url_sonar = "https://github.com/professortiagoinfnet/inteligencia_artificial/blob/main/sonar_dataset.csv"
df = baixar_csv_github(url_sonar)

X = df.iloc[:, :-1]
y = preprocessing.LabelEncoder().fit_transform(df.iloc[:, -1])
 
treino_a, teste_a, treino_c, teste_c = train_test_split(X, y, test_size=0.2, random_state=42)