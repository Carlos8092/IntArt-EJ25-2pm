import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import StandardScaler

# Mostrar gráficas
plt.show()

# Cargar datos
dataframe = pd.read_csv(r"usuarios_win_mac_lin.csv")

# Análisis descriptivo
print(dataframe.head())
print(dataframe.describe())
print(dataframe.groupby('clase').size())  # Analizar cuántos resultados hay por clase

# Graficar histogramas
dataframe.drop(['clase'], axis=1).hist()
plt.show()

# Pairplot (actualizando parámetro 'size' a 'height')
sb.pairplot(dataframe.dropna(), hue='clase', height=4, vars=["duracion", "paginas", "acciones", "valor"], kind='reg')

# Preparar datos para el modelo
X = np.array(dataframe.drop(['clase'], axis=1))  
y = np.array(dataframe['clase'])

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) 

# Entrenar el modelo
model = linear_model.LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# Realizar predicciones y mostrar los primeros 5
predictions = model.predict(X_scaled)
print(predictions[0:5])

# Validación y evaluación
model.score(X_scaled, y)
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X_scaled, y, test_size=validation_size, random_state=seed)

# Validación cruzada
kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
msg = "%s: %f (%f)" % ('Logistic Regression', cv_results.mean(), cv_results.std())
print(msg)

# Evaluar en datos de validación
predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# Realizar predicción para nuevos datos
X_new = pd.DataFrame({'duracion': [10], 'paginas': [3], 'acciones': [5], 'valor': [9]})
X_new_scaled = scaler.transform(X_new)  # Escalar los nuevos datos
print(model.predict(X_new_scaled))  # Realizar la predicción

