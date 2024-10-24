import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Cargar los datos desde el archivo Excel
file_path = r'D:\InteligenciaNegocios\Semana10\Data10-1.xlsx'
data = pd.read_excel(file_path)

X = data[['Precio actual', 'Precio final']]
y = data['Estado'].apply(lambda x: 1 if x == 'Alto' else 0)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Estandarizar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear el clasificador SVM
svm = SVC(kernel='linear', C=1.0, random_state=42)

# Entrenar el clasificador
svm.fit(X_train, y_train)

# Hacer predicciones
y_pred = svm.predict(X_test)

# Evaluar el desempeño
accuracy = np.mean(y_pred == y_test)
print(f"Precisión del modelo: {accuracy:.2f}")

# Visualizar los resultados
def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = plt.cm.RdYlBu

    # Representar la superficie de decisión
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Representar los ejemplos por clase
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=[colors[idx]],
                    marker=markers[idx], label=cl, edgecolor='black')

# Convertir X_test a numpy array para la función de visualización
X_test_np = np.array(X_test)

plot_decision_regions(X_test_np, y_test.values, classifier=svm)
plt.xlabel('Precio actual [estandarizado]')
plt.ylabel('Precio final [estandarizado]')
plt.legend(loc='upper left')
plt.show()
