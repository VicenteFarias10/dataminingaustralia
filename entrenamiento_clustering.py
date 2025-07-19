import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("Cargando datos...")
df = pd.read_csv('../data/wheaterPba3Completo.csv')

print("Preparando datos para clustering...")
# Seleccionar variables para clustering
features = ['MinTemp', 'MaxTemp', 'Humidity9am', 'Humidity3pm', 'Rainfall', 'Evaporation']
df_clustering = df[features].dropna()

print(f"Datos disponibles para clustering: {len(df_clustering)} registros")

# Escalar los datos
print("Escalando datos...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clustering)

# Entrenar modelo K-Means
print("Entrenando modelo K-Means...")
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Guardar modelo y scaler
print("Guardando modelo de clustering...")
joblib.dump(kmeans, '../models/clustering/modelo_entrenado.pkl')
joblib.dump(scaler, '../models/clustering/scaler.pkl')

print("Modelo de clustering guardado como '../models/clustering/modelo_entrenado.pkl'")
print("Scaler guardado como '../models/clustering/scaler.pkl'")

# Crear visualización de ejemplo
print("Creando visualización de ejemplo...")
plt.figure(figsize=(12, 8))

# Gráfico de dispersión de temperatura vs humedad
plt.subplot(2, 2, 1)
colors = ['blue', 'green']
for i in range(2):
    mask = clusters == i
    plt.scatter(df_clustering[mask]['MaxTemp'], 
               df_clustering[mask]['Humidity3pm'], 
               c=colors[i], label=f'Cluster {i}', alpha=0.7)
plt.xlabel('Temperatura Máxima (°C)')
plt.ylabel('Humedad 3pm (%)')
plt.title('Clustering: Temperatura vs Humedad')
plt.legend()

# Gráfico de distribución de clusters
plt.subplot(2, 2, 2)
cluster_counts = pd.Series(clusters).value_counts().sort_index()
plt.bar(cluster_counts.index, cluster_counts.values, color=['blue', 'green'])
plt.xlabel('Cluster')
plt.ylabel('Número de Registros')
plt.title('Distribución de Clusters')
plt.xticks([0, 1])

# Gráfico de características por cluster
plt.subplot(2, 2, 3)
df_clustering['cluster'] = clusters
cluster_means = df_clustering.groupby('cluster')[features].mean()
cluster_means.T.plot(kind='bar', ax=plt.gca())
plt.title('Características Promedio por Cluster')
plt.xticks(rotation=45)
plt.legend(['Cluster 0', 'Cluster 1'])

# Gráfico de correlación
plt.subplot(2, 2, 4)
correlation_matrix = df_clustering[features].corr()
plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xticks(range(len(features)), features, rotation=45)
plt.yticks(range(len(features)), features)
plt.title('Matriz de Correlación')

plt.tight_layout()
plt.savefig('../models/clustering/analisis_clustering.png', dpi=300, bbox_inches='tight')
print("Visualización guardada como '../models/clustering/analisis_clustering.png'")
plt.close()

print("\n--- Entrenamiento de clustering completado ---")
print("Archivos generados:")
print("- ../models/clustering/modelo_entrenado.pkl: Modelo entrenado")
print("- ../models/clustering/scaler.pkl: Scaler")
print("- ../models/clustering/analisis_clustering.png: Visualización de ejemplo") 