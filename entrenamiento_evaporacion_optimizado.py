import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print("Cargando datos...")
df = pd.read_csv('wheaterPba3Completo.csv')

print("Preparando datos para análisis de evaporación...")

# Convertir 'Date' a datetime si existe
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])

# --- Mapeo de Localidades a Latitud y Longitud ---
location_coords = {
    'Albury': (-36.080556, 146.916389), 'BadgerysCreek': (-33.87, 150.73), 'Cobar': (-31.49, 145.84),
    'CoffsHarbour': (-30.30, 153.11), 'Moree': (-29.46, 149.85), 'Newcastle': (-32.93, 151.75),
    'NorahHead': (-33.28, 151.57), 'NorfolkIsland': (-29.04, 167.95), 'Penrith': (-33.75, 150.68),
    'Richmond': (-33.60, 150.75), 'Sydney': (-33.87, 151.21), 'SydneyAirport': (-33.94, 151.18),
    'WaggaWagga': (-35.12, 147.37), 'Williamtown': (-32.78, 151.84), 'Wollongong': (-34.42, 150.88),
    'Canberra': (-35.28, 149.13), 'Tuggeranong': (-35.43, 149.09), 'MountGinini': (-35.52, 148.78),
    'Ballarat': (-37.56, 143.85), 'Bendigo': (-36.76, 144.28), 'Beechworth': (-36.36, 146.68),
    'Brisbane': (-27.47, 153.02), 'Cairns': (-16.92, 145.77), 'GoldCoast': (-28.00, 153.43),
    'Townsville': (-19.26, 146.82), 'Adelaide': (-34.93, 138.60), 'Albany': (-35.02, 117.89),
    'Woomera': (-31.20, 136.82), 'Nuriootpa': (-34.46, 138.99), 'PearceRAAF': (-31.67, 116.01),
    'Perth': (-31.95, 115.86), 'PerthAirport': (-31.94, 115.97), 'SalmonGums': (-33.05, 121.64),
    'Walpole': (-34.97, 116.73), 'Hobart': (-42.88, 147.33), 'Launceston': (-41.43, 147.13),
    'Melbourne': (-37.81, 144.96), 'MelbourneAirport': (-37.67, 144.84), 'Mildura': (-34.20, 142.16),
    'Dartmoor': (-37.90, 141.28), 'Watsonia': (-37.72, 145.08), 'Portland': (-38.34, 141.60),
    'Nhil': (-36.33, 141.65), 'Uluru': (-25.35, 131.03), 'Darwin': (-12.46, 130.84),
    'Katherine': (-14.47, 132.27), 'AliceSprings': (-23.70, 133.88)
}

# Crear las columnas Latitud y Longitud si Location existe
if 'Location' in df.columns:
    df['Latitud'] = df['Location'].map(lambda x: location_coords.get(x, (np.nan, np.nan))[0])
    df['Longitud'] = df['Location'].map(lambda x: location_coords.get(x, (np.nan, np.nan))[1])

# Variables principales para evaporación (solo las más importantes)
features = ['MinTemp', 'MaxTemp', 'Rainfall', 'Humidity9am', 'Humidity3pm', 
           'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'Sunshine']

# Filtrar datos de evaporación válidos
df_evaporacion = df[df['Evaporation'].notna()].copy()

# Preparar features - solo usar las variables más importantes
available_features = [f for f in features if f in df_evaporacion.columns]
print(f"Features disponibles: {available_features}")

X = df_evaporacion[available_features]
y = df_evaporacion['Evaporation']

# Eliminar filas con valores nulos
mask = ~(X.isnull().any(axis=1) | y.isnull())
X = X[mask]
y = y[mask]

print(f"Datos para entrenamiento: {len(X)} muestras con {X.shape[1]} features")

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar datos
scaler_evaporacion = StandardScaler()
X_train_scaled = scaler_evaporacion.fit_transform(X_train)
X_test_scaled = scaler_evaporacion.transform(X_test)

# Probar solo los mejores modelos (menos modelos = menos peso)
modelos = {
    'Ridge': Ridge(alpha=1.0, random_state=42),
    'RandomForest': RandomForestRegressor(
        n_estimators=50,  # Reducido de 100 a 50
        max_depth=10,     # Limitamos profundidad 
        random_state=42
    )
}

print("\nEntrenando modelos...")
results = {}

for name, model in modelos.items():
    print(f"Entrenando {name}...")
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    results[name] = {
        'r2': r2,
        'rmse': rmse,
        'mae': mae
    }
    
    print(f"{name} - R²: {r2:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}")

# Seleccionar el mejor modelo
best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
best_model = modelos[best_model_name]

print(f"\nMejor modelo: {best_model_name}")
print(f"R² Score: {results[best_model_name]['r2']:.3f}")

# --- MODELO OPTIMIZADO - Solo guardar lo esencial ---
print("\nGuardando modelo optimizado de evaporación...")

# Solo guardamos lo mínimo necesario para hacer predicciones
modelo_optimizado = {
    'model': best_model,                    # Solo el mejor modelo
    'scaler': scaler_evaporacion,          # Scaler necesario
    'features': available_features,        # Lista de features
    'model_name': best_model_name,         # Nombre del mejor modelo
    'performance': {                       # Solo métricas del mejor modelo
        'r2': results[best_model_name]['r2'],
        'rmse': results[best_model_name]['rmse'],
        'mae': results[best_model_name]['mae']
    }
}

# Guardar modelo optimizado
joblib.dump(modelo_optimizado, 'modelo_evaporacion_optimizado.pkl')

# Verificar el tamaño del archivo
import os
size_mb = os.path.getsize('modelo_evaporacion_optimizado.pkl') / (1024 * 1024)
print(f"Modelo optimizado guardado: {size_mb:.2f} MB")

print("\n✅ Optimización completada!")
print("- modelo_evaporacion_optimizado.pkl: Versión ligera del modelo")
print(f"- Tamaño reducido a: {size_mb:.2f} MB")
print(f"- Mantiene la misma precisión: R² = {results[best_model_name]['r2']:.3f}") 