import pandas as pd
import requests
from io import StringIO
import matplotlib.pyplot as plt
from sklearn import preprocessing, decomposition, tree, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split


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

X = df.iloc[:, :-1]
y = preprocessing.LabelEncoder().fit_transform(df.iloc[:, -1])
 
  