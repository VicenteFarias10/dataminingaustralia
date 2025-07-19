import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix

print("Cargando datos...")
df = pd.read_csv('wheaterPba3Completo.csv')

print("Preparando datos para predicción de lluvia...")

# Convertir 'Date' a datetime si existe
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])

# --- 1. Crear la variable RainToday ---
# RainToday: 1 si Rainfall > 0 mm, 0 si no. Manejar NaNs en Rainfall.
df['RainToday'] = df['Rainfall'].apply(lambda x: 1 if x > 0 else (0 if x == 0 else np.nan))

# --- Seleccionar las columnas relevantes para este análisis ---
selected_cols = [
    'Pressure9am', 'Pressure3pm', 'Humidity9am', 'Humidity3pm',
    'MaxTemp', 'WindGustSpeed', 'RainToday'
]

# Verificar qué columnas existen en el dataset
available_cols = [col for col in selected_cols if col in df.columns]
print(f"Columnas disponibles: {available_cols}")

# Agregar RainTomorrow si existe, si no, crear basado en Rainfall
if 'RainTomorrow' in df.columns:
    df_rain_prediction = df[available_cols + ['RainTomorrow']].copy()
else:
    # Crear RainTomorrow basado en Rainfall del día siguiente
    # Para simplificar, usaremos Rainfall > 0 como proxy
    df['RainTomorrow'] = (df['Rainfall'] > 0).astype(int)
    df_rain_prediction = df[available_cols + ['RainTomorrow']].copy()

# --- Manejo de Valores Faltantes (Imputación y Eliminación) ---

# Variable objetivo: RainTomorrow
print(f"Filas iniciales antes de eliminar NaNs en RainTomorrow: {df_rain_prediction.shape[0]}")
df_rain_prediction.dropna(subset=['RainTomorrow'], inplace=True)
print(f"Filas después de eliminar NaNs en RainTomorrow: {df_rain_prediction.shape[0]}")

# Variables Predictoras: Imputar columnas numéricas con su mediana
numerical_cols_to_impute = [col for col in available_cols if col != 'RainToday']
for col in numerical_cols_to_impute:
    if df_rain_prediction[col].isnull().any():
        median_val = df_rain_prediction[col].median()
        df_rain_prediction[col].fillna(median_val, inplace=True)
        print(f"NaNs en '{col}' imputados con la mediana: {median_val:.2f}")

# Imputar NaNs en 'RainToday' con la moda (0 o 1)
if 'RainToday' in df_rain_prediction.columns and df_rain_prediction['RainToday'].isnull().any():
    mode_raintoday = df_rain_prediction['RainToday'].mode()[0]
    df_rain_prediction['RainToday'].fillna(mode_raintoday, inplace=True)
    print(f"NaNs en 'RainToday' imputados con la moda: {int(mode_raintoday)}")

# --- Codificar RainTomorrow si es necesario ---
if df_rain_prediction['RainTomorrow'].dtype == 'object':
    df_rain_prediction['RainTomorrow'] = df_rain_prediction['RainTomorrow'].map({'Yes': 1, 'No': 0})

print("\n--- Resumen de la Preparación de Datos ---")
print(f"Dimensiones del DataFrame final: {df_rain_prediction.shape}")
print("\nConteo de valores faltantes después de la preparación:")
print(df_rain_prediction.isnull().sum())
print("\nConteo de valores de RainTomorrow (0=No, 1=Yes):")
print(df_rain_prediction['RainTomorrow'].value_counts())

# --- Análisis Exploratorio de Datos ---
print("\n--- Análisis Exploratorio de Datos ---")

# 1. Distribución de la variable objetivo
plt.figure(figsize=(7, 5))
sns.countplot(x='RainTomorrow', data=df_rain_prediction)
plt.title('Distribución de RainTomorrow (0=No Lluvia, 1=Lluvia)', fontsize=16)
plt.xlabel('RainTomorrow', fontsize=12)
plt.ylabel('Conteo', fontsize=12)
plt.xticks([0, 1], ['No Lluvia', 'Sí Lluvia'])
plt.savefig('distribucion_lluvia.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Distribución de variables numéricas por RainTomorrow
numerical_features = [col for col in available_cols if col != 'RainToday']
if len(numerical_features) > 0:
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(numerical_features):
        plt.subplot(2, 3, i + 1)
        sns.boxplot(x='RainTomorrow', y=feature, data=df_rain_prediction)
        plt.title(f'{feature} por RainTomorrow', fontsize=14)
        plt.xlabel('RainTomorrow (0=No, 1=Yes)', fontsize=10)
        plt.ylabel(feature, fontsize=10)
        plt.xticks([0, 1], ['No Lluvia', 'Sí Lluvia'])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('boxplots_lluvia.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Relación de RainToday con RainTomorrow (si existe)
if 'RainToday' in df_rain_prediction.columns:
    plt.figure(figsize=(7, 5))
    sns.countplot(x='RainToday', hue='RainTomorrow', data=df_rain_prediction)
    plt.title('Relación entre RainToday y RainTomorrow', fontsize=16)
    plt.xlabel('RainToday (0=No Llovió Hoy, 1=Sí Llovió Hoy)', fontsize=12)
    plt.ylabel('Conteo', fontsize=12)
    plt.xticks([0, 1], ['No Llovió Hoy', 'Sí Llovió Hoy'])
    plt.legend(title='RainTomorrow', labels=['No Lloverá Mañana', 'Sí Lloverá Mañana'])
    plt.savefig('relacion_raintoday.png', dpi=300, bbox_inches='tight')
    plt.close()

# 4. Matriz de Correlación
print("\n--- Matriz de Correlación ---")
correlation_matrix = df_rain_prediction.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matriz de Correlación de Variables para Predicción de Lluvia', fontsize=16)
plt.savefig('correlacion_lluvia.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Preparación para el modelo ---
# Separar features y target
features = [col for col in available_cols if col != 'RainTomorrow']
X = df_rain_prediction[features]
y = df_rain_prediction['RainTomorrow']

print(f"\nFeatures para el modelo: {features}")
print(f"Shape de X: {X.shape}")
print(f"Shape de y: {y.shape}")

# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train set: {len(X_train)} registros")
print(f"Test set: {len(X_test)} registros")

# Escalar features
scaler_lluvia = StandardScaler()
X_train_scaled = scaler_lluvia.fit_transform(X_train)
X_test_scaled = scaler_lluvia.transform(X_test)

# --- Entrenar modelo Logistic Regression ---
print("\nEntrenando modelo Logistic Regression...")
modelo_lluvia = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced'
)

modelo_lluvia.fit(X_train_scaled, y_train)

# --- Evaluar modelo ---
y_pred = modelo_lluvia.predict(X_test_scaled)
y_pred_proba = modelo_lluvia.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nMétricas del modelo:")
print(f"Accuracy: {accuracy:.3f}")
print(f"ROC AUC: {roc_auc:.3f}")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# --- Guardar modelo ---
print("\nGuardando modelo de predicción de lluvia...")
modelo_completo_lluvia = {
    'model': modelo_lluvia,
    'scaler': scaler_lluvia,
    'features': features,
    'accuracy': accuracy,
    'roc_auc': roc_auc
}

joblib.dump(modelo_completo_lluvia, 'modelo_lluvia.pkl')
print("Modelo de lluvia guardado como 'modelo_lluvia.pkl'")

# --- Visualizaciones adicionales ---
plt.figure(figsize=(15, 10))

# Matriz de confusión
plt.subplot(2, 3, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.ylabel('Real')
plt.xlabel('Predicción')

# Curva ROC
plt.subplot(2, 3, 2)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC')
plt.legend(loc="lower right")

# Distribución de probabilidades
plt.subplot(2, 3, 3)
plt.hist(y_pred_proba, bins=20, alpha=0.7)
plt.xlabel('Probabilidad de lluvia')
plt.ylabel('Frecuencia')
plt.title('Distribución de probabilidades')

# Importancia de features (coeficientes)
plt.subplot(2, 3, 4)
coef_df = pd.DataFrame({
    'feature': features,
    'coefficient': modelo_lluvia.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

plt.barh(range(len(coef_df)), coef_df['coefficient'])
plt.yticks(range(len(coef_df)), coef_df['feature'])
plt.xlabel('Coeficiente')
plt.title('Importancia de Features (Coeficientes)')
plt.gca().invert_yaxis()

# Predicciones vs reales
plt.subplot(2, 3, 5)
plt.scatter(y_test, y_pred_proba, alpha=0.6)
plt.xlabel('Real (0=No, 1=Sí)')
plt.ylabel('Probabilidad Predicha')
plt.title('Predicciones vs Valores Reales')

plt.tight_layout()
plt.savefig('analisis_lluvia_completo.png', dpi=300, bbox_inches='tight')
print("Visualización completa guardada como 'analisis_lluvia_completo.png'")
plt.close()

print("\n--- Entrenamiento de predicción de lluvia completado ---")
print("Archivos generados:")
print("- modelo_lluvia.pkl: Modelo entrenado")
print("- distribucion_lluvia.png: Distribución de la variable objetivo")
print("- boxplots_lluvia.png: Distribución de variables por clase")
print("- relacion_raintoday.png: Relación RainToday vs RainTomorrow")
print("- correlacion_lluvia.png: Matriz de correlación")
print("- analisis_lluvia_completo.png: Análisis completo del modelo")
print(f"- Accuracy: {accuracy:.3f}")
print(f"- ROC AUC: {roc_auc:.3f}") 