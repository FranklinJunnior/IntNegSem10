import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Cargar el archivo Excel
file_path = "D:/InteligenciaNegocios/Semana10/Data10-1.xlsx"  # Cambia la ruta según tu archivo

df = pd.read_excel(file_path)

# Selección de las características para el agrupamiento
X = df[['Precio actual', 'Precio final']]

# Aplicar el algoritmo K-Means
kmeans = KMeans(n_clusters=3, random_state=42)  # Definimos 3 grupos (puedes cambiar este valor)
df['Cluster'] = kmeans.fit_predict(X)

# Visualización de los clusters
plt.scatter(df['Precio actual'], df['Precio final'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Precio actual')
plt.ylabel('Precio final')
plt.title('Clusters usando K-Means')
plt.show()

# Imprimir los centroides
print("Centroides de los clusters:")
print(kmeans.cluster_centers_)
