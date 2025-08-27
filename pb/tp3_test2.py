import pandas as pd
import requests
from io import StringIO
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection, decomposition, tree, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

# ==============================
# 0. Download dos dados Sonar
# ==============================
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

# ==============================
# 1. PCA aplicado ao KNN
# ==============================
pca = decomposition.PCA(n_components=0.95)  # mantém 95% da variância
X_pca = pca.fit_transform(X)
ptreino_a, pteste_a, ptreino_c, pteste_c = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# ============================== 
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
    def imprimir_resultados(nome_modelo, teste_c, previstos):
        print(f"\nResultados do modelo: {nome_modelo}")

        acuracia, precisao, recall, f1score = ClassificaBinario.calcula_precisao_recall_f1score(teste_c, previstos)
        _, especificidade, _ = ClassificaBinario.calcula_acuracia_especificidade(teste_c, previstos)

        print(f"Acurácia: {acuracia:.4f}")
        print(f"Precisão: {precisao:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1score:.4f}")
        print(f"Especificidade: {especificidade:.4f}")

# ==============================
# 1. PCA aplicado ao KNN
# ==============================
pca = decomposition.PCA(n_components=0.95)  # mantém 95% da variância
X_pca = pca.fit_transform(X)

ptreino_a, pteste_a, ptreino_c, pteste_c = train_test_split(X_pca, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(ptreino_a, ptreino_c)
previstos = knn.predict(pteste_a)

print("\n=== Avaliação do KNN com PCA (95% variância) ===")


acuracia, precisao, recall, f1score = ClassificaBinario.calcula_precisao_recall_f1score(teste_c, previstos)
_, especificidade, _ = ClassificaBinario.calcula_acuracia_especificidade(teste_c, previstos)

print("Acurácia       :", round(acuracia, 4))
print("Precisão      :", round(precisao, 4))
print("Recall        :", round(recall, 4))
print("F1-Score      :", round(f1score, 4))
print("Especificidade:", round(especificidade, 4))

probabilidades = knn.predict_proba(teste_a)[:, 1]
ClassificaBinario.plotar_roc(teste_c, probabilidades, titulo="Curva ROC - KNN com PCA")

# ==============================
# 2. Árvore de Decisão
# ==============================
treino_a, teste_a, treino_c, teste_c = train_test_split(X, y, test_size=0.2, random_state=42)

decision_tree = tree.DecisionTreeClassifier(random_state=42)
decision_tree.fit(treino_a, treino_c)
previstos = decision_tree.predict(teste_a)

print("\n=== Avaliação da Árvore de Decisão (sem ajuste) ===")
acuracia, precisao, recall, f1score = ClassificaBinario.calcula_precisao_recall_f1score(teste_c, previstos)
_, especificidade, _ = ClassificaBinario.calcula_acuracia_especificidade(teste_c, previstos)

print("Acurácia       :", round(acuracia, 4))
print("Precisão      :", round(precisao, 4))
print("Recall        :", round(recall, 4))
print("F1-Score      :", round(f1score, 4))
print("Especificidade:", round(especificidade, 4))

probabilidades = decision_tree.predict_proba(teste_a)[:, 1]
ClassificaBinario.plotar_roc(teste_c, probabilidades, titulo="Curva ROC - Árvore de Decisão")

# ==============================
# 3. GridSearch na Árvore de Decisão
# ==============================
param_grid = {
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=tree.DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X, y)

print("\n=== Melhores hiperparâmetros encontrados ===")
print(grid_search.best_params_)
print("Melhor score:", grid_search.best_score_)
#asd


arvore_ajustada = tree.DecisionTreeClassifier(**best_params, random_state=42)
arvore_ajustada.fit(treino_a, treino_c)
previstos = arvore_ajustada.predict(teste_a)

print("\n=== Avaliação da Árvore de Decisão (ajustada pelo GridSearch) ===")
acuracia, precisao, recall, f1score = ClassificaBinario.calcula_precisao_recall_f1score(teste_c, previstos)
_, especificidade, _ = ClassificaBinario.calcula_acuracia_especificidade(teste_c, previstos)

print("Acurácia       :", round(acuracia, 4))
print("Precisão      :", round(precisao, 4))
print("Recall        :", round(recall, 4))
print("F1-Score      :", round(f1score, 4))
print("Especificidade:", round(especificidade, 4))

probabilidades = arvore_ajustada.predict_proba(teste_a)[:, 1]
ClassificaBinario.plotar_roc(teste_c, probabilidades, titulo="Curva ROC - Árvore Ajustada (GridSearch)")