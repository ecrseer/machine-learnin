# Importações básicas
import numpy as np
import pandas as pd
# Machine Learning
from sklearn import datasets, tree, model_selection
# Visualização
import matplotlib.pyplot as plt

# Funções
def classificar (decision_tree, atributos, target, param_grid):
  grid_search = model_selection.GridSearchCV(decision_tree, param_grid, cv=5)
  grid_search.fit (atributos, target)
  results = grid_search.cv_results_
  best_estimator = grid_search.best_estimator_
  return results, best_estimator

def imprimir_resultados (results, best_estimator, msg=""):
  print (f"Tempo médio de Teste     {msg}: {results['mean_score_time']}")
  print (f"Acurácia média de Teste  {msg}: {results['mean_test_score']}")
  print (f"Profundidade máxima      {msg}: {best_estimator.tree_.max_depth}")

def gerar_coords (atributos, best_estimator):
  y_min = atributos[:, 1].min()
  y_max = atributos[:, 1].max()

  x_min = atributos[:, 0].min()
  x_max = atributos[:, 0].max()

  feature = best_estimator.tree_.feature
  threshold = best_estimator.tree_.threshold

  coords = []
  for i in range (len(feature)):
    if feature[i] == 0:
      # traçar uma linha vertical
      x0 = threshold[i]
      y0 = y_min
      x1 = threshold[i]
      y1 = y_max
      coords.append (((x0, x1), (y0, y1)))
    elif feature[i] == 1:
      # traçar uma linha horizontal
      x0 = x_min
      y0 = threshold[i]
      x1 = x_max
      y1 = threshold[i]
      coords.append (((x0, x1), (y0, y1)))
  return coords

# Código Principal (main)
atributos, target = datasets.make_blobs (n_features=2, centers=2, n_samples=100, cluster_std=[10, 10])

# MODELO BASELINE (SEM PODA)
param_grid = {'criterion': ['gini']}
results, best_estimator = classificar (tree.DecisionTreeClassifier(), atributos, target, param_grid)

# MODELO COM PODA
param_grid2 = {"max_depth": [2, 4, 6, 8, 10, 12]}
results2, best_estimator2 = classificar (tree.DecisionTreeClassifier(), atributos, target, param_grid2)

#IMPRIMIR RESULTADOS
imprimir_resultados(results, best_estimator, 'BASELINE')
imprimir_resultados(results2, best_estimator2, 'COM PODA')

#GERAR COORDENADAS
coords = gerar_coords(atributos, best_estimator)
coords2 = gerar_coords(atributos, best_estimator2)

# y_min = atributos[:, 1].min()
# y_max = atributos[:, 1].max()

# x_min = atributos[:, 0].min()
# x_max = atributos[:, 0].max()

# feature = best_estimator.tree_.feature
# threshold = best_estimator.tree_.threshold

# coords = []
# for i in range (len(feature)):
#   if feature[i] == 0:
#     # traçar uma linha vertical
#     x0 = threshold[i]
#     y0 = y_min
#     x1 = threshold[i]
#     y1 = y_max
#     coords.append (((x0, x1), (y0, y1)))
#   elif feature[i] == 1:
#     # traçar uma linha horizontal
#     x0 = x_min
#     y0 = threshold[i]
#     x1 = x_max
#     y1 = threshold[i]
#     coords.append (((x0, x1), (y0, y1)))

#print (f"coords: {coords}")

#GERAR GRÁFICO
plt.scatter (atributos[:, 0][target==0], atributos[:, 1][target==0], color='blue')
plt.scatter (atributos[:, 0][target==1], atributos[:, 1][target==1], color='red')
for coord in coords:
  plt.plot (coord[0], coord[1], color='black')
plt.show()

plt.scatter (atributos[:, 0][target==0], atributos[:, 1][target==0], color='blue')
plt.scatter (atributos[:, 0][target==1], atributos[:, 1][target==1], color='red')
for coord in coords2:
  plt.plot (coord[0], coord[1], color='black')
plt.show()