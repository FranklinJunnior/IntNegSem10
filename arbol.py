import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt

# Cargar el archivo Excel
file_path = "D:/InteligenciaNegocios/Semana10/Data10-1.xlsx" 
df = pd.read_excel(file_path)

# Selección de las características y la columna objetivo
X = df[['Precio actual', 'Precio final']]
y = df['Estado']

# División de los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de Árbol de Decisión
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predicción
y_pred = clf.predict(X_test)

# Precisión del modelo
accuracy = clf.score(X_test, y_test)
print(f"Precisión del modelo de Árbol de Decisión: {accuracy * 100:.2f}%")

# Visualización del Árbol de Decisión
plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=['Precio actual', 'Precio final'], class_names=df['Estado'].unique(), filled=True)
plt.title('Árbol de Decisión para predecir Estado')
plt.show()
