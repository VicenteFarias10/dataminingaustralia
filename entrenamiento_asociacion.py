import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Cargar los datos
print("Cargando datos...")
df = pd.read_csv('wheaterPba3Completo.csv')

# Limpiar y preparar datos
print("Preparando datos...")
df_imputed = df.copy()

# Variables a discretizar
variables_a_discretizar = ['MaxTemp', 'Humidity3pm', 'WindGustSpeed', 'Sunshine']
num_bins = 3
labels = ['Baja', 'Media', 'Alta']

print("\n--- Discretización de Variables Climáticas ---")

# Aplicar discretización a cada variable
for var in variables_a_discretizar:
    if var in df_imputed.columns:
        if var == 'Sunshine':
            # Para Sunshine, usamos pd.cut con rangos de igual ancho
            min_val = df_imputed[var].min()
            max_val = df_imputed[var].max()
            bins = np.linspace(min_val, max_val, num_bins + 1)
            bins[-1] = bins[-1] + 0.001
            try:
                df_imputed[f'{var}_Category'] = pd.cut(df_imputed[var], bins=bins, labels=labels, include_lowest=True, right=True)
                print(f"Variable '{var}' discretizada en '{var}_Category' usando pd.cut.")
            except Exception as e:
                print(f"Error al discretizar '{var}': {e}")
        else:
            # Para las otras variables, usamos pd.qcut
            try:
                df_imputed[f'{var}_Category'] = pd.qcut(df_imputed[var], q=num_bins, labels=labels, duplicates='drop')
                print(f"Variable '{var}' discretizada en '{var}_Category' usando pd.qcut.")
            except Exception as e:
                print(f"Error al discretizar '{var}': {e}")

# Crear encoders para las categorías
print("\n--- Creando encoders para las categorías ---")
encoders = {}
category_cols = [f'{v}_Category' for v in variables_a_discretizar if f'{v}_Category' in df_imputed.columns]

for col in category_cols:
    if col in df_imputed.columns:
        le = LabelEncoder()
        # Asegurar que no hay valores nulos
        df_imputed[col] = df_imputed[col].fillna('Baja')
        le.fit(df_imputed[col])
        encoders[col] = le
        print(f"Encoder creado para {col}")

# Identificar patrones frecuentes
print("\n--- Identificación de Patrones Frecuentes ---")
if category_cols:
    frequent_combinations = df_imputed[category_cols].value_counts().reset_index(name='Count')
    
    # Crear columna de texto legible
    frequent_combinations['Combination_String'] = frequent_combinations.apply(
        lambda row: ', '.join([f"{col.replace('_Category', '')}: {row[col]}" for col in category_cols]), axis=1
    )
    
    print(f"Se encontraron {len(frequent_combinations)} combinaciones únicas")
    print("Top 5 combinaciones más frecuentes:")
    print(frequent_combinations.head())

# Guardar el modelo y encoders
print("\n--- Guardando modelo y encoders ---")
modelo_asociacion = {
    'encoders': encoders,
    'variables_discretizadas': variables_a_discretizar,
    'category_cols': category_cols,
    'labels': labels,
    'frequent_combinations': frequent_combinations if 'frequent_combinations' in locals() else None
}

joblib.dump(modelo_asociacion, 'modelo_asociacion.pkl')
print("Modelo de asociación guardado como 'modelo_asociacion.pkl'")

# Crear visualización de ejemplo
print("\n--- Creando visualización de ejemplo ---")
if 'frequent_combinations' in locals() and len(frequent_combinations) > 0:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(14, 8))
    top_10 = frequent_combinations.head(10)
    
    # Crear gráfico de barras
    plt.bar(range(len(top_10)), top_10['Count'], color='skyblue', edgecolor='navy')
    plt.title('Top 10 Combinaciones de Condiciones Climáticas Frecuentes', fontsize=16)
    plt.xlabel('Combinación de Condiciones Climáticas', fontsize=12)
    plt.ylabel('Frecuencia de Ocurrencia', fontsize=12)
    
    # Configurar etiquetas del eje X
    plt.xticks(range(len(top_10)), [f"Combinación {i+1}" for i in range(len(top_10))], rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Guardar la figura
    plt.savefig('patrones_frecuentes_asociacion.png', dpi=300, bbox_inches='tight')
    print("Visualización guardada como 'patrones_frecuentes_asociacion.png'")
    plt.close()

print("\n--- Entrenamiento completado ---")
print("Archivos generados:")
print("- modelo_asociacion.pkl: Modelo entrenado")
print("- patrones_frecuentes_asociacion.png: Visualización de ejemplo") 