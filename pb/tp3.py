import pandas as pd
import numpy as np
import requests
from io import StringIO
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection, neighbors, metrics
from sklearn.feature_extraction import text


# questao 1:Criação das features:
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
 