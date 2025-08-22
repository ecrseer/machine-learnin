import pandas as pd
import numpy as np

from sklearn import preprocessing, model_selection, neighbors, metrics
from sklearn.feature_extraction import text


def obter_dataset (url, columns_names):
  df = pd.read_csv (url, sep="\t", header=None, names=columns_names)
  df[columns_names[0]] = preprocessing.LabelEncoder().fit_transform(df[columns_names[0]])
  return df

def vetorizar (vetorizador, atributo):
  atributo_vec = vetorizador.fit_transform (atributo)
  atributo_vec_df = pd.DataFrame (atributo_vec.toarray(), columns=vetorizador.get_feature_names_out())
  return atributo_vec_df

def classificar (modelo, atributos, classe):
  treino_a, teste_a, treino_c, teste_c = model_selection.train_test_split (atributos, classe, train_size=0.8)
  modelo.fit (treino_a, treino_c)
  previstos = modelo.predict (teste_a)
  acuracia = metrics.accuracy_score (previstos, teste_c)
  return acuracia


### MAIN ####

URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
NOMES_COLUNAS = ['classe', 'mensagem']

df = obter_dataset (URL, NOMES_COLUNAS)

#Definição do modelo
knn = neighbors.KNeighborsClassifier (n_neighbors=10)
knns = [neighbors.KNeighborsClassifier (n_neighbors=k) for k in [10, 100, 500, 1000, 1500, 2000]]

# Definição BoW (atributos, classe)
atributos_bow = vetorizar (text.CountVectorizer(), df['mensagem'])
classe_bow = df['classe']

# # Definição TF-IDF (atributos, classe)
atributos_tfidf = vetorizar (text.TfidfVectorizer (), df['mensagem'])
classe_tfidf = df['classe']

# Classificação
res_bow = model_selection.cross_validate (knn, atributos_bow, classe_bow, return_train_score=True)
res_tfidf = model_selection.cross_validate (knn, atributos_tfidf, classe_tfidf, return_train_score=True)

# Exibição de Resultados
print ("Resultado com BoW     : ", res_bow['test_score'].mean ())
print ("Resultados com TF-IDF : ", res_tfidf['test_score'].mean ())
