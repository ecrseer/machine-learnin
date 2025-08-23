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


def obter_dataset_noticias():
    url_true = "https://github.com/professortiagoinfnet/inteligencia_artificial/blob/main/True.csv"
    url_fake = "https://github.com/professortiagoinfnet/inteligencia_artificial/blob/main/Fake.csv"
    
    df_true = baixar_csv_github(url_true)
    df_fake = baixar_csv_github(url_fake)
    
    primeira_coluna_true = df_true.columns[0]
    primeira_coluna_fake = df_fake.columns[0]
    
    df_true_clean = pd.DataFrame({
        'classe': 1,
        'mensagem': df_true[primeira_coluna_true]
    })
    
    df_fake_clean = pd.DataFrame({
        'classe': 0,
        'mensagem': df_fake[primeira_coluna_fake]
    })
    
    df_combined = pd.concat([df_true_clean, df_fake_clean], ignore_index=True)
    df_combined = df_combined.dropna()
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    df_combined['classe'] = preprocessing.LabelEncoder().fit_transform(df_combined['classe'])
    
    return df_combined


def vetorizar(vetorizador, atributo):
    atributo_vec = vetorizador.fit_transform(atributo)
    atributo_vec_df = pd.DataFrame(atributo_vec.toarray(), columns=vetorizador.get_feature_names_out())
    return atributo_vec_df


df = obter_dataset_noticias()

print("Definindo os modelos KNN...")
# questao 2: Modelagem de K-Nearest Neighbors (KNN): 
knns = [neighbors.KNeighborsClassifier(n_neighbors=k) for k in [1, 2, 3,7,10,14,30,42]]

print("bag of words (BoW)")
atributos_bow = vetorizar(text.CountVectorizer(), df['mensagem'])
classe_bow = df['classe']

print("TF-IDF")
atributos_tfidf = vetorizar(text.TfidfVectorizer(), df['mensagem'])
classe_tfidf = df['classe']


def classificar(modelo, atributos, classe):
    treino_a, teste_a, treino_c, teste_c = model_selection.train_test_split(
        atributos, classe, train_size=0.8
    )
    modelo.fit(treino_a, treino_c)
    previstos = modelo.predict(teste_a)
    return treino_a, teste_a, treino_c, teste_c, previstos, modelo

# questao 4: Avaliação de Classificadores Binários:
class ClassificaBinario:
    @staticmethod
    def calcula_precisao_recall_f1score(y_true, y_pred):
        acuracia = metrics.accuracy_score(y_true, y_pred)
        precisao = metrics.precision_score(y_true, y_pred, zero_division=0)
        recall = metrics.recall_score(y_true, y_pred)
        f1score = metrics.f1_score(y_true, y_pred)
        return acuracia, precisao, recall, f1score

    @staticmethod
    def calcula_acuracia_especificidade(y_true, y_pred):
        acuracia = metrics.accuracy_score(y_true, y_pred)
        mc = metrics.confusion_matrix(y_true, y_pred)
        ((TN, FP), (FN, TP)) = mc
        especificidade = TN / (TN + FP)
        sensibilidade = TP / (TP + FN)
        return acuracia, especificidade, sensibilidade

    @staticmethod
    def plotar_roc(y_true, y_prob, titulo="Curva ROC"):
        fpr, tpr, _ = metrics.roc_curve(y_true, y_prob)
        roc_auc = metrics.auc(fpr, tpr)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("Falso Positivo")
        plt.ylabel("Verdadeiro Positivo")
        plt.title(titulo)
        plt.legend(loc="lower right")
        plt.show()



for modelo_knn in knns:
    print(f"\n\n=== Avaliando KNN com k={modelo_knn.n_neighbors} ===")
    treino_a, teste_a, treino_c, teste_c, previstos, modelo = classificar(
        modelo_knn, atributos_tfidf, classe_tfidf
    )

    acuracia, precisao, recall, f1score = ClassificaBinario.calcula_precisao_recall_f1score(teste_c, previstos)
    _, especificidade, _recal = ClassificaBinario.calcula_acuracia_especificidade(teste_c, previstos)

    print("Acurácia       :", round(acuracia, 4))
    print("Precisão      :", round(precisao, 4))
    print("Recall        :", round(recall, 4))
    print("F1-Score      :", round(f1score, 4))
    print("Especificidade:", round(especificidade, 4))

    probabilidades = modelo.predict_proba(teste_a)[:, 1]
    ClassificaBinario.plotar_roc(teste_c, probabilidades, titulo=f"Curva ROC (k={modelo_knn.n_neighbors})")

#Questao 5: Baseado nos valores encontrados para as diferentes figuras de mérito,
#  interprete os resultados e disserte sobre a eficiência do classificador criado.
