import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("Cargando datos...")
df = pd.read_csv('wheaterPba3Completo.csv')

print("Preparando datos para análisis de confort...")
# Seleccionar variables relevantes para el confort
features = ['MinTemp', 'MaxTemp', 'Humidity3pm', 'WindGustSpeed', 'Sunshine', 'Rainfall', 'Latitud', 'Longitud']
df_confort = df[['Location'] + features].dropna()

print(f"Datos disponibles: {len(df_confort)} registros")

# Agrupar por ubicación para obtener promedios
print("Agrupando datos por ubicación...")
df_grouped = df_confort.groupby('Location').agg({
    'MinTemp': 'mean',
    'MaxTemp': 'mean',
    'Humidity3pm': 'mean',
    'WindGustSpeed': 'mean',
    'Sunshine': 'mean',
    'Rainfall': 'mean',
    'Latitud': 'mean',
    'Longitud': 'mean'
}).reset_index()

print(f"Ubicaciones únicas: {len(df_grouped)}")

# Renombrar columnas para consistencia
df_grouped.rename(columns={
    'MinTemp': 'Avg_MinTemp',
    'MaxTemp': 'Avg_MaxTemp',
    'Humidity3pm': 'Avg_Humidity3pm',
    'WindGustSpeed': 'Avg_WindGustSpeed',
    'Sunshine': 'Avg_Sunshine',
    'Rainfall': 'Avg_Rainfall'
}, inplace=True)

# Variables para el modelo
expected_features = ['Avg_MinTemp', 'Avg_MaxTemp', 'Avg_Humidity3pm',
                     'Avg_WindGustSpeed', 'Avg_Sunshine', 'Avg_Rainfall']

print("Entrenando modelo K-Means para confort...")
# Preparar datos para clustering
X = df_grouped[expected_features]

# Escalar los datos
scaler_confort = StandardScaler()
X_scaled = scaler_confort.fit_transform(X)

# Entrenar modelo K-Means con 3 clusters
kmeans_confort = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans_confort.fit_predict(X_scaled)

# Analizar clusters para asignar etiquetas de confort
print("Analizando clusters para asignar etiquetas de confort...")
df_grouped['cluster'] = clusters

# Analizar características de cada cluster
cluster_analysis = df_grouped.groupby('cluster')[expected_features].mean()
print("\nCaracterísticas promedio por cluster:")
print(cluster_analysis)

# Asignar etiquetas basadas en el análisis
# Cluster con condiciones más extremas será "NO CONFORME"
cluster_labels = {}

# Identificar el cluster menos confortable (temperaturas extremas, alta humedad, etc.)
for cluster_id in range(3):
    cluster_data = cluster_analysis.loc[cluster_id]
    
    # Calcular score de confort (valores más altos = menos confortable)
    temp_score = abs(cluster_data['Avg_MaxTemp'] - 25)  # Temperatura ideal ~25°C
    humidity_score = cluster_data['Avg_Humidity3pm']  # Humedad alta = menos confortable
    wind_score = cluster_data['Avg_WindGustSpeed']  # Viento fuerte = menos confortable
    
    comfort_score = temp_score + humidity_score * 0.1 + wind_score * 0.01
    
    if comfort_score == cluster_analysis.apply(lambda x: abs(x['Avg_MaxTemp'] - 25) + x['Avg_Humidity3pm'] * 0.1 + x['Avg_WindGustSpeed'] * 0.01, axis=1).max():
        cluster_labels[cluster_id] = "NO CONFORME"
    else:
        cluster_labels[cluster_id] = "CONFORME"

print("\nEtiquetas asignadas por cluster:")
for cluster_id, label in cluster_labels.items():
    print(f"Cluster {cluster_id}: {label}")

# Aplicar etiquetas
df_grouped['Etiqueta_Confort'] = df_grouped['cluster'].map(cluster_labels)

# Guardar modelo y componentes
print("\nGuardando modelo de confort...")
modelo_confort = {
    'kmeans_model': kmeans_confort,
    'scaler': scaler_confort,
    'cluster_labels': cluster_labels,
    'expected_features': expected_features,
    'df_grouped': df_grouped
}

joblib.dump(modelo_confort, 'modelo_confort.pkl')
print("Modelo de confort guardado como 'modelo_confort.pkl'")

# Mostrar estadísticas
print("\nEstadísticas del análisis de confort:")
confort_stats = df_grouped['Etiqueta_Confort'].value_counts()
print(confort_stats)

print(f"\nPorcentaje de zonas no confortables: {(confort_stats.get('NO CONFORME', 0) / len(df_grouped) * 100):.1f}%")

# Crear visualización de ejemplo
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

# Gráfico de barras por etiqueta de confort
plt.subplot(2, 2, 1)
confort_stats.plot(kind='bar', color=['green', 'red'])
plt.title('Distribución de Confort por Zona')
plt.ylabel('Número de Zonas')
plt.xticks(rotation=45)

# Gráfico de dispersión de temperatura vs humedad
plt.subplot(2, 2, 2)
colors = {'CONFORME': 'green', 'NO CONFORME': 'red'}
for confort in ['CONFORME', 'NO CONFORME']:
    mask = df_grouped['Etiqueta_Confort'] == confort
    plt.scatter(df_grouped[mask]['Avg_MaxTemp'], 
               df_grouped[mask]['Avg_Humidity3pm'], 
               c=colors[confort], label=confort, alpha=0.7)
plt.xlabel('Temperatura Máxima Promedio (°C)')
plt.ylabel('Humedad 3pm Promedio (%)')
plt.title('Temperatura vs Humedad por Confort')
plt.legend()

# Gráfico de distribución de temperaturas
plt.subplot(2, 2, 3)
for confort in ['CONFORME', 'NO CONFORME']:
    mask = df_grouped['Etiqueta_Confort'] == confort
    plt.hist(df_grouped[mask]['Avg_MaxTemp'], alpha=0.7, label=confort, bins=10)
plt.xlabel('Temperatura Máxima Promedio (°C)')
plt.ylabel('Frecuencia')
plt.title('Distribución de Temperaturas por Confort')
plt.legend()

# Gráfico de distribución de humedad
plt.subplot(2, 2, 4)
for confort in ['CONFORME', 'NO CONFORME']:
    mask = df_grouped['Etiqueta_Confort'] == confort
    plt.hist(df_grouped[mask]['Avg_Humidity3pm'], alpha=0.7, label=confort, bins=10)
plt.xlabel('Humedad 3pm Promedio (%)')
plt.ylabel('Frecuencia')
plt.title('Distribución de Humedad por Confort')
plt.legend()

plt.tight_layout()
plt.savefig('analisis_confort.png', dpi=300, bbox_inches='tight')
print("Visualización guardada como 'analisis_confort.png'")
plt.close()

print("\n--- Entrenamiento de confort completado ---")
print("Archivos generados:")
print("- modelo_confort.pkl: Modelo entrenado")
print("- analisis_confort.png: Visualización de ejemplo") 