import pandas as pd
import numpy as np
import requests
from io import StringIO

from sklearn import preprocessing, model_selection, neighbors, metrics
from sklearn.feature_extraction import text


def baixar_csv_github(url):
    """Baixa um CSV do GitHub e retorna um DataFrame"""
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


def obter_dataset_noticias():
    """Cria um dataset combinado com notícias verdadeiras e falsas"""
    # URLs dos arquivos
    url_true = "https://github.com/professortiagoinfnet/inteligencia_artificial/blob/main/True.csv"
    url_fake = "https://github.com/professortiagoinfnet/inteligencia_artificial/blob/main/Fake.csv"
    
    # Baixa os datasets
    df_true = baixar_csv_github(url_true)
    df_fake = baixar_csv_github(url_fake)
    
    # Extrai a primeira coluna
    primeira_coluna_true = df_true.columns[0]
    primeira_coluna_fake = df_fake.columns[0]
    
    # Cria DataFrames com texto e classe
    df_true_clean = pd.DataFrame({
        'classe': 1,  # 1 para verdadeira
        'mensagem': df_true[primeira_coluna_true]
    })
    
    df_fake_clean = pd.DataFrame({
        'classe': 0,  # 0 para falsa
        'mensagem': df_fake[primeira_coluna_fake]
    })
    
    # Combina os datasets
    df_combined = pd.concat([df_true_clean, df_fake_clean], ignore_index=True)
    df_combined = df_combined.dropna()
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Codifica as classes como no código original
    df_combined['classe'] = preprocessing.LabelEncoder().fit_transform(df_combined['classe'])
    
    return df_combined


def vetorizar(vetorizador, atributo):
    atributo_vec = vetorizador.fit_transform(atributo)
    atributo_vec_df = pd.DataFrame(atributo_vec.toarray(), columns=vetorizador.get_feature_names_out())
    return atributo_vec_df


def classificar(modelo, atributos, classe):
    treino_a, teste_a, treino_c, teste_c = model_selection.train_test_split(atributos, classe, train_size=0.8)
    modelo.fit(treino_a, treino_c)
    previstos = modelo.predict(teste_a)
    acuracia = metrics.accuracy_score(previstos, teste_c)
    return acuracia


### MAIN ####

# Obter dataset de notícias
df = obter_dataset_noticias()

print("Definindo os modelos KNN...")
knn = neighbors.KNeighborsClassifier(n_neighbors=10)
knns = [neighbors.KNeighborsClassifier(n_neighbors=k) for k in [10,50, 100]]

print("bag of words (BoW)")
atributos_bow = vetorizar(text.CountVectorizer(), df['mensagem'])
classe_bow = df['classe']

print("TF-IDF")
atributos_tfidf = vetorizar(text.TfidfVectorizer(), df['mensagem'])
classe_tfidf = df['classe']



def calcula_acuracia_especificidade(y_true, y_pred):
    acuracia = metrics.accuracy_score(y_true, y_pred)
    mc = metrics.confusion_matrix(y_true, y_pred)
    ((TN, FP), (FN, TP)) = mc
    especificidade = TN / (TN + FP)
    return (acuracia, especificidade)


def calcula_precisao_recall(y_true, y_pred):
    acuracia = metrics.accuracy_score(y_true, y_pred)
    precisao = metrics.precision_score(y_true, y_pred, zero_division=0)
    recall = metrics.recall_score(y_true, y_pred)
    return (acuracia, precisao, recall)


def calcula_metricas_completas(y_true, y_pred):
    acuracia = metrics.accuracy_score(y_true, y_pred)
    precisao = metrics.precision_score(y_true, y_pred, zero_division=0)
    recall = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    
    # Matriz de confusão para especificidade
    mc = metrics.confusion_matrix(y_true, y_pred)
    ((TN, FP), (FN, TP)) = mc
    especificidade = TN / (TN + FP)
    sensibilidade = recall  # Sensibilidade é igual ao recall
    
    return (acuracia, precisao, recall, f1, sensibilidade, especificidade)


treino_a_tfidf, teste_a_tfidf, treino_c_tfidf, teste_c_tfidf = model_selection.train_test_split(
    atributos_tfidf, classe_tfidf, train_size=0.8, random_state=42)

print("Avaliaca com TF-IDF:")
for modelo in knns:
    k = modelo.n_neighbors
    modelo.fit(treino_a_tfidf, treino_c_tfidf)
    previstos_tfidf = modelo.predict(teste_a_tfidf)
    
    acuracia, precisao, recall, f1, sensibilidade, especificidade = calcula_metricas_completas(
        teste_c_tfidf, previstos_tfidf)
    
    print(f"\nK={k} - TF-IDF:")
    print(f"  Acurácia      : {acuracia:.4f}")
    print(f"  Precisão      : {precisao:.4f}")
    print(f"  Recall        : {recall:.4f}")
    print(f"  F1-Score      : {f1:.4f}")
    print(f"  Sensibilidade : {sensibilidade:.4f}")
    print(f"  Especificidade: {especificidade:.4f}")

# Curva ROC para o melhor modelo (K=10)
print("\n" + "="*80)
print("CURVA ROC - Modelo KNN com K=10")


print("usando validação cruzada")
res_bow = model_selection.cross_validate(knn, atributos_bow, classe_bow, return_train_score=True)
res_tfidf = model_selection.cross_validate(knn, atributos_tfidf, classe_tfidf, return_train_score=True)


print("Resultado com BoW     : ", res_bow['test_score'].mean())
print("Resultados com TF-IDF : ", res_tfidf['test_score'].mean())

# Testando diferentes valores de K
print("\nTestando diferentes valores de K:")
for modelo in knns:
    k = modelo.n_neighbors
    res_bow_k = model_selection.cross_validate(modelo, atributos_bow, classe_bow)
    res_tfidf_k = model_selection.cross_validate(modelo, atributos_tfidf, classe_tfidf)
    
    print(f"K={k} - BoW: {res_bow_k['test_score'].mean():.4f}, TF-IDF: {res_tfidf_k['test_score'].mean():.4f}")