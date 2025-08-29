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

url_sonar = "https://github.com/professortiagoinfnet/inteligencia_artificial/blob/main/sonar_dataset.csv"
df = baixar_csv_github(url_sonar)

X = df.iloc[:, :-1]
y = preprocessing.LabelEncoder().fit_transform(df.iloc[:, -1])
 
treino_a, teste_a, treino_c, teste_c = train_test_split(X, y, test_size=0.2, random_state=42)

# 
# questao 1. PCA aplicado ao KNN
#  
pca = decomposition.PCA(n_components=0.95)  
ptreino_a = pca.fit_transform(treino_a)
pteste_a = pca.transform(teste_a) 
ptreino_c = treino_c
pteste_c = teste_c
 
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

    @staticmethod
    def imprimir_resultados(nome_modelo, teste_c, previstos):
        print(f"\nResultados do modelo: {nome_modelo}")
        acuracia, precisao, recall, f1score = ClassificaBinario.calcula_precisao_recall_f1score(teste_c, previstos)
        _, especificidade, sensibilidade = ClassificaBinario.calcula_acuracia_especificidade(teste_c, previstos)

        print(f"Acurácia       : {acuracia:.4f}")
        print(f"Precisão       : {precisao:.4f}")
        print(f"Recall         : {recall:.4f}")
        print(f"F1-Score       : {f1score:.4f}")
        print(f"Especificidade : {especificidade:.4f}")
        print(f"Sensibilidade  : {sensibilidade:.4f}")

 


knnSemPca = KNeighborsClassifier(n_neighbors=5)
knnSemPca.fit(treino_a, treino_c)
previstos = knnSemPca.predict(teste_a)

print("\n Avaliação do KNN sem PCA")
ClassificaBinario.imprimir_resultados("KNN sem pca",teste_c,previstos)
probabilidades = knnSemPca.predict_proba(teste_a)[:, 1]
ClassificaBinario.plotar_roc(teste_c, probabilidades, titulo="Curva ROc KNN sem pca")

 
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(ptreino_a, ptreino_c)
previstos = knn.predict(pteste_a)

print("\n Avaliação do KNN com PCA (95% variância) ")
ClassificaBinario.imprimir_resultados("KNN com pca",pteste_c,previstos)
probabilidades = knn.predict_proba(pteste_a)[:, 1]
ClassificaBinario.plotar_roc(pteste_c, probabilidades, titulo="Curva ROc KNN com PCA")

# 
# questao 2. Árvore de Decisão  

decision_tree = tree.DecisionTreeClassifier(random_state=42)
decision_tree.fit(treino_a, treino_c)
previstos = decision_tree.predict(teste_a)

ClassificaBinario.imprimir_resultados("Arvore decisao",teste_c,previstos)

probabilidades = decision_tree.predict_proba(teste_a)[:, 1]
ClassificaBinario.plotar_roc(teste_c, probabilidades, titulo="Curva ROC - Árvore de Decisão")

# 
# questao 4. Busca Hiperparamétrica: 
# Utilizar GridSearch para otimizar os hiperparâmetros dos modelos.

param_grid = {
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# questao 5: Pruning de Árvores de Decisão
# poda com param_grid.max_depth
#
grid_search = GridSearchCV(
    estimator=tree.DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
grid_search.fit(treino_a, treino_c)
print("\n Melhores hiperparâmetros encontrados ")
print(grid_search.best_params_)
print("Melhor score (CV):", grid_search.best_score_)


best_params = grid_search.best_params_
arvore_ajustada = tree.DecisionTreeClassifier(**best_params, random_state=42)
arvore_ajustada.fit(treino_a, treino_c)
previstos = arvore_ajustada.predict(teste_a)

print("\n Avaliação da Árvore de Decisão  pelo GridSearch ")

# questao 6 Avaliação de Classificadores Binários: 

ClassificaBinario.imprimir_resultados("Árvore de decisao com gridsearch",teste_c, previstos)

probabilidades = arvore_ajustada.predict_proba(teste_a)[:, 1]
ClassificaBinario.plotar_roc(teste_c, probabilidades, titulo="Curva ROC - Árvore Ajustada (GridSearch)")


## questao 7 Baseado nos valores encontrados para as 
# diferentes figuras de mérito, interprete os resultados e disserte 
# sobre a eficiência do classificador criado.
"""
Baseando-se nas métricas obtidas, o KNN com PCA perdeu em Precisão
 que é em relaçao aos falsos positivos, nessa métrica
 quanto mais falsos positivos pior será a precisão, assim
 sendo Precisão é um indicativo que diz:
dos positivos encontrados, quais realmente eram verdadeiramente positivos?
 e no caso do código, caiu de 61% de precisão para 57%

Já em relação à Àrvore de decisão tivemos um ganho enorme no Recall
 que é em uma relação que usa os falsos negativos:
 quanto mais falsos negativos significa que o recall deixou de encontrar positivos,
 assim sendo recall diz: dos positivos encontrados, quantos foram acertados?
 e no caso da Árvore de decisão passamos de 66% para 80%
Todas as outras métricas também melhoraram, sendo o Recall a mais expressiva
"""