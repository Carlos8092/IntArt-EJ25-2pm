import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from pylab import rcParams
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedBaggingClassifier
from collections import Counter

# Cargar los datos
df = pd.read_csv("creditcard.csv") 
df.head(n=5)

# Imprimir la forma del dataframe y las frecuencias de las clases
print(df.shape)
count_classes = pd.Series(df['Class']).value_counts(sort=True)
print(count_classes)

# Definir etiquetas para el gráfico
LABELS = ['Class 0', 'Class 1']

# Graficar la distribución de clases
count_classes.plot(kind='bar', rot=0)
plt.xticks(range(2), LABELS)
plt.title("Frequency by observation number")
plt.xlabel("Class")
plt.ylabel("Number of Observations")
plt.show()

# Definir las características y las etiquetas
y = df['Class']
X = df.drop('Class', axis=1)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

# Crear una función para entrenar el modelo
def run_model(X_train, X_test, y_train, y_test):
    clf_base = LogisticRegression(C=1.0, penalty='l2', random_state=1, solver="newton-cg")
    clf_base.fit(X_train, y_train)
    return clf_base

# Ejecutar el modelo base
model = run_model(X_train, X_test, y_train, y_test)

# Definir una función para mostrar los resultados
def mostrar_resultados(y_test, pred_y):
    conf_matrix = confusion_matrix(y_test, pred_y)
    plt.figure(figsize=(12, 12))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
    print(classification_report(y_test, pred_y))

# Predicción con el modelo base
pred_y = model.predict(X_test)
mostrar_resultados(y_test, pred_y)

# Crear el modelo balanceado
def run_model_balanced(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(C=1.0, penalty='l2', random_state=1, solver="newton-cg", class_weight="balanced")
    clf.fit(X_train, y_train)
    return clf

# Ejecutar el modelo balanceado
model = run_model_balanced(X_train, X_test, y_train, y_test)
pred_y = model.predict(X_test)
mostrar_resultados(y_test, pred_y)

# Aplicar NearMiss (undersampling)
us = NearMiss(ratio=0.5, n_neighbors=3, version=2, random_state=1)
X_train_res, y_train_res = us.fit_resample(X_train, y_train)

print("Distribution before resampling {}".format(Counter(y_train)))
print("Distribution after resampling {}".format(Counter(y_train_res)))

# Ejecutar el modelo con el dataset balanceado por NearMiss
model = run_model(X_train_res, X_test, y_train_res, y_test)
pred_y = model.predict(X_test)
mostrar_resultados(y_test, pred_y)

# Aplicar RandomOverSampler (oversampling)
os = RandomOverSampler(sampling_strategy=0.5)
X_train_res, y_train_res = os.fit_resample(X_train, y_train)

print("Distribution before resampling {}".format(Counter(y_train)))
print("Distribution after resampling {}".format(Counter(y_train_res)))

# Ejecutar el modelo con el dataset balanceado por RandomOverSampler
model = run_model(X_train_res, X_test, y_train_res, y_test)
pred_y = model.predict(X_test)
mostrar_resultados(y_test, pred_y)

# Aplicar SMOTETomek (combinación de SMOTE y undersampling)
os_us = SMOTETomek(sampling_strategy=0.5)
X_train_res, y_train_res = os_us.fit_resample(X_train, y_train)

print("Distribution before resampling {}".format(Counter(y_train)))
print("Distribution after resampling {}".format(Counter(y_train_res)))

# Ejecutar el modelo con el dataset balanceado por SMOTETomek
model = run_model(X_train_res, X_test, y_train_res, y_test)
pred_y = model.predict(X_test)
mostrar_resultados(y_test, pred_y)

# Crear el BalancedBaggingClassifier
bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                sampling_strategy='auto', replacement=False, random_state=0)

# Entrenar el clasificador
bbc.fit(X_train, y_train)
pred_y = bbc.predict(X_test)
mostrar_resultados(y_test, pred_y)
from sklearn.ensemble import RandomForestClassifier

 # Crear el modelo con 100 arboles
model = RandomForestClassifier(n_estimators=100,
bootstrap = True, verbose=2,
max_features = 'sqrt')
# a entrenar!
model.fit(X_train, y_train)