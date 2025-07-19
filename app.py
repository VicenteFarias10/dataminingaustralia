from flask import Flask, jsonify, render_template, request
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Cargar modelos
try:
    modelo_clustering = joblib.load('modelo_entrenado.pkl')
    scaler_clustering = joblib.load('scaler.pkl')
    print("Modelo de clustering cargado correctamente")
except:
    modelo_clustering = None
    scaler_clustering = None
    print("Modelo de clustering no encontrado")

# Cargar modelo de asociación (si existe)
try:
    modelo_asociacion = joblib.load('modelo_asociacion.pkl')
    print("Modelo de asociación cargado correctamente")
except:
    modelo_asociacion = None
    print("Modelo de asociación no encontrado")

# Cargar modelo de confort (si existe)
try:
    modelo_confort = joblib.load('modelo_confort.pkl')
    print("Modelo de confort cargado correctamente")
except:
    modelo_confort = None
    print("Modelo de confort no encontrado")

# Cargar modelo de lluvia (si existe)
try:
    modelo_lluvia = joblib.load('modelo_lluvia.pkl')
    print("Modelo de lluvia cargado correctamente")
except:
    modelo_lluvia = None
    print("Modelo de lluvia no encontrado")

# Cargar modelo de evaporación (si existe)
try:
    modelo_evaporacion = joblib.load('modelo_evaporacion_optimizado.pkl')
    print("Modelo de evaporación optimizado cargado correctamente")
except:
    modelo_evaporacion = None
    print("Modelo de evaporación optimizado no encontrado")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/data', methods=['POST'])
def data():
    try:
        input_json = request.get_json()
        
        # Extraer técnica y datos
        technique = input_json.get('technique', 'clustering')
        data = input_json.get('data', [])
        
        if not data:
            return jsonify({'error': 'No se proporcionaron datos'}), 400
        
        df_input = pd.DataFrame(data)
        
        # Procesar según la técnica seleccionada
        if technique == 'clustering':
            return procesar_clustering(df_input)
        elif technique == 'confort':
            return procesar_confort(df_input)
        elif technique == 'lluvia':
            return procesar_lluvia(df_input)
        elif technique == 'evaporacion':
            return procesar_evaporacion(df_input)
        elif technique == 'asociacion':
            return procesar_asociacion(df_input)
        else:
            return jsonify({'error': f'Técnica no soportada: {technique}'}), 400
            
    except Exception as e:
        print(f"Error en /data POST: {e}")
        return jsonify({'error': str(e)}), 400

def procesar_clustering(df_input):
    """Procesar datos para clustering (técnica actual)"""
    if modelo_clustering is None or scaler_clustering is None:
        raise ValueError("Modelo de clustering no está disponible.")
    
    expected_columns = [
        'MinTemp', 'MaxTemp', 'Humidity9am', 'Humidity3pm',
        'Rainfall', 'Evaporation', 'Latitud', 'Longitud'
    ]

    for col in expected_columns:
        if col not in df_input.columns:
            raise ValueError(f"Falta la columna requerida: {col}")

    df_model = df_input[expected_columns]
    X = scaler_clustering.transform(df_model)
    clusters = modelo_clustering.predict(X)

    df_input['cluster'] = clusters
    df_input.rename(columns={'Latitud': 'latitude', 'Longitud': 'longitude'}, inplace=True)

    if 'Location' not in df_input.columns:
        df_input['Location'] = ['Lugar ' + str(i) for i in range(len(df_input))]

    df_response = df_input[['Location', 'latitude', 'longitude', 'cluster']]
    return jsonify(df_response.to_dict(orient='records'))

def procesar_confort(df_input):
    """Procesar datos para análisis de confort"""
    if modelo_confort is None:
        raise ValueError("Modelo de confort no está disponible. Ejecuta primero el entrenamiento.")
    
    if 'Latitud' in df_input.columns and 'Longitud' in df_input.columns:
        df_input.rename(columns={'Latitud': 'latitude', 'Longitud': 'longitude'}, inplace=True)
        
        # Obtener componentes del modelo
        kmeans_model = modelo_confort['kmeans_model']
        scaler = modelo_confort['scaler']
        cluster_labels = modelo_confort['cluster_labels']
        expected_features = modelo_confort['expected_features']
        
        # Preparar datos para el modelo
        df_processed = df_input.copy()
        
        # Mapear columnas de entrada a las esperadas por el modelo
        column_mapping = {
            'MinTemp': 'Avg_MinTemp',
            'MaxTemp': 'Avg_MaxTemp', 
            'Humidity3pm': 'Avg_Humidity3pm',
            'WindGustSpeed': 'Avg_WindGustSpeed',
            'Sunshine': 'Avg_Sunshine',
            'Rainfall': 'Avg_Rainfall'
        }
        
        # Crear columnas esperadas por el modelo
        for input_col, model_col in column_mapping.items():
            if input_col in df_processed.columns:
                df_processed[model_col] = df_processed[input_col]
            else:
                # Si no existe la columna, usar valores por defecto
                if 'MinTemp' in input_col or 'MaxTemp' in input_col:
                    df_processed[model_col] = 25.0  # Temperatura moderada
                elif 'Humidity' in input_col:
                    df_processed[model_col] = 60.0  # Humedad moderada
                elif 'Wind' in input_col:
                    df_processed[model_col] = 30.0  # Viento moderado
                elif 'Sunshine' in input_col:
                    df_processed[model_col] = 8.0   # Horas de sol moderadas
                elif 'Rainfall' in input_col:
                    df_processed[model_col] = 2.0   # Lluvia moderada
        
        # Preparar datos para predicción
        X = df_processed[expected_features]
        X_scaled = scaler.transform(X)
        
        # Predecir clusters
        clusters = kmeans_model.predict(X_scaled)
        
        # Asignar etiquetas de confort
        etiquetas_confort = [cluster_labels.get(c, "DESCONOCIDO") for c in clusters]
        
        # Calcular score de confort (0-100, donde 100 es muy confortable)
        confort_scores = []
        for i, row in df_processed.iterrows():
            temp_score = max(0, 100 - abs(row['Avg_MaxTemp'] - 25) * 2)  # Temperatura ideal ~25°C
            humidity_score = max(0, 100 - row['Avg_Humidity3pm'] * 0.5)  # Humedad alta = menos confortable
            wind_score = max(0, 100 - row['Avg_WindGustSpeed'] * 0.1)    # Viento fuerte = menos confortable
            
            # Score promedio ponderado
            score = (temp_score * 0.5 + humidity_score * 0.3 + wind_score * 0.2)
            confort_scores.append(int(score))
        
        # Preparar respuesta
        if 'Location' not in df_processed.columns:
            df_processed['Location'] = ['Lugar ' + str(i) for i in range(len(df_processed))]
        
        response_data = []
        for i, row in df_processed.iterrows():
            response_data.append({
                'Location': row['Location'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'etiqueta_confort': etiquetas_confort[i],
                'confort_score': confort_scores[i],
                'cluster': int(clusters[i]),
                'MaxTemp': row.get('Avg_MaxTemp', 'N/A'),
                'Humidity3pm': row.get('Avg_Humidity3pm', 'N/A'),
                'WindGustSpeed': row.get('Avg_WindGustSpeed', 'N/A')
            })
        
        return jsonify(response_data)
    else:
        raise ValueError("Se requieren columnas Latitud y Longitud para el análisis de confort")

def procesar_lluvia(df_input):
    """Procesar datos para predicción de lluvia"""
    if modelo_lluvia is None:
        raise ValueError("Modelo de lluvia no está disponible. Ejecuta primero el entrenamiento.")
    
    try:
        if 'Latitud' in df_input.columns and 'Longitud' in df_input.columns:
            df_input.rename(columns={'Latitud': 'latitude', 'Longitud': 'longitude'}, inplace=True)
            
            # Obtener componentes del modelo
            model = modelo_lluvia['model']
            scaler = modelo_lluvia['scaler']
            features = modelo_lluvia['features']
            
            print(f"Features requeridas para lluvia: {features}")
            print(f"Columnas disponibles: {list(df_input.columns)}")
            
            # Preparar datos para predicción
            df_processed = df_input.copy()
            
            # Convertir RainToday de texto a número si es necesario
            if 'RainToday' in df_processed.columns:
                if df_processed['RainToday'].dtype == 'object':
                    df_processed['RainToday'] = df_processed['RainToday'].map({'Yes': 1, 'No': 0})
            
            # Crear columnas faltantes con valores por defecto
            for feature in features:
                if feature not in df_processed.columns:
                    if 'Temp' in feature:
                        df_processed[feature] = 25.0  # Temperatura moderada
                    elif 'Humidity' in feature:
                        df_processed[feature] = 60.0  # Humedad moderada
                    elif 'Wind' in feature:
                        df_processed[feature] = 30.0  # Viento moderado
                    elif 'Pressure' in feature:
                        df_processed[feature] = 1013.0  # Presión estándar
                    elif 'Cloud' in feature:
                        df_processed[feature] = 5.0  # Nubosidad moderada
                    elif 'Sunshine' in feature:
                        df_processed[feature] = 8.0  # Horas de sol moderadas
            
            # Preparar datos para el modelo
            X = df_processed[features]
            X_scaled = scaler.transform(X)
            
            # Predecir probabilidades
            probabilidades = model.predict_proba(X_scaled)[:, 1]
            predicciones = model.predict(X_scaled)
            
            # Preparar respuesta
            if 'Location' not in df_processed.columns:
                df_processed['Location'] = ['Lugar ' + str(i) for i in range(len(df_processed))]
            
            response_data = []
            for i, row in df_processed.iterrows():
                response_data.append({
                    'Location': row['Location'],
                    'latitude': row['latitude'],
                    'longitude': row['longitude'],
                    'probabilidad_lluvia': float(probabilidades[i]),
                    'llovera_manana': bool(predicciones[i]),
                    'MaxTemp': row.get('MaxTemp', 'N/A'),
                    'Humidity3pm': row.get('Humidity3pm', 'N/A'),
                    'WindGustSpeed': row.get('WindGustSpeed', 'N/A')
                })
            
            return jsonify(response_data)
        else:
            raise ValueError("Se requieren columnas Latitud y Longitud para la predicción de lluvia")
    except Exception as e:
        print(f"Error en procesar_lluvia: {e}")
        raise ValueError(f"Error procesando datos de lluvia: {str(e)}")

def procesar_evaporacion(df_input):
    """Procesar datos para análisis de evaporación"""
    if modelo_evaporacion is None:
        raise ValueError("Modelo de evaporación no está disponible. Ejecuta primero el entrenamiento.")
    
    try:
        if 'Latitud' in df_input.columns and 'Longitud' in df_input.columns:
            df_input.rename(columns={'Latitud': 'latitude', 'Longitud': 'longitude'}, inplace=True)
            
            # Obtener componentes del modelo
            model = modelo_evaporacion['model']
            scaler = modelo_evaporacion['scaler']
            features = modelo_evaporacion['features']
            
            print(f"Features requeridas para evaporación: {features}")
            print(f"Columnas disponibles: {list(df_input.columns)}")
            
            # Mapear columnas si es necesario
            df_processed = df_input.copy()
            if 'latitude' in df_processed.columns and 'longitude' in df_processed.columns:
                df_processed['Latitud'] = df_processed['latitude']
                df_processed['Longitud'] = df_processed['longitude']
            
            # Función para categorizar evaporación (si no está en el modelo)
            def categorizar_evaporacion(valor):
                if valor < 3:
                    return 'Baja'
                elif valor < 6:
                    return 'Media'
                else:
                    return 'Alta'
            
            # Preparar datos para predicción
            # df_processed ya está creado arriba
            
            # Crear columnas faltantes con valores por defecto
            for feature in features:
                if feature not in df_processed.columns:
                    if 'Temp' in feature:
                        df_processed[feature] = 25.0  # Temperatura moderada
                    elif 'Humidity' in feature:
                        df_processed[feature] = 60.0  # Humedad moderada
                    elif 'Wind' in feature:
                        df_processed[feature] = 30.0  # Viento moderado
                    elif 'Pressure' in feature:
                        df_processed[feature] = 1013.0  # Presión estándar
                    elif 'Cloud' in feature:
                        df_processed[feature] = 5.0  # Nubosidad moderada
                    elif 'Sunshine' in feature:
                        df_processed[feature] = 8.0  # Horas de sol moderadas
            
            # Preparar datos para el modelo
            X = df_processed[features]
            X_scaled = scaler.transform(X)
            
            # Predecir evaporación
            evaporacion_predicha = model.predict(X_scaled)
            
            # Categorizar evaporación
            categorias = [categorizar_evaporacion(pred) for pred in evaporacion_predicha]
            
            # Preparar respuesta
            if 'Location' not in df_processed.columns:
                df_processed['Location'] = ['Lugar ' + str(i) for i in range(len(df_processed))]
            
            response_data = []
            for i, row in df_processed.iterrows():
                response_data.append({
                    'Location': row['Location'],
                    'latitude': row['latitude'],
                    'longitude': row['longitude'],
                    'evaporacion_predicha': float(evaporacion_predicha[i]),
                    'nivel_evaporacion': categorias[i],
                    'MaxTemp': row.get('MaxTemp', 'N/A'),
                    'Humidity3pm': row.get('Humidity3pm', 'N/A'),
                    'WindGustSpeed': row.get('WindGustSpeed', 'N/A'),
                    'Sunshine': row.get('Sunshine', 'N/A')
                })
            
            return jsonify(response_data)
        else:
            raise ValueError("Se requieren columnas Latitud y Longitud para el análisis de evaporación")
    except Exception as e:
        print(f"Error en procesar_evaporacion: {e}")
        raise ValueError(f"Error procesando datos de evaporación: {str(e)}")

def procesar_asociacion(df_input):
    """Procesar datos para análisis de asociación"""
    if modelo_asociacion is None:
        raise ValueError("Modelo de asociación no está disponible. Ejecuta primero el entrenamiento.")
    
    if 'Latitud' in df_input.columns and 'Longitud' in df_input.columns:
        df_input.rename(columns={'Latitud': 'latitude', 'Longitud': 'longitude'}, inplace=True)
        
        # Obtener variables del modelo
        variables_discretizadas = modelo_asociacion['variables_discretizadas']
        encoders = modelo_asociacion['encoders']
        labels = modelo_asociacion['labels']
        frequent_combinations = modelo_asociacion['frequent_combinations']
        
        # Discretizar variables para los datos de entrada
        df_processed = df_input.copy()
        
        for var in variables_discretizadas:
            if var in df_processed.columns:
                try:
                    # Usar los mismos bins que se usaron en el entrenamiento
                    df_processed[f'{var}_Category'] = pd.qcut(df_processed[var], q=3, labels=labels, duplicates='drop')
                except:
                    # Si falla, usar cut
                    min_val = df_processed[var].min()
                    max_val = df_processed[var].max()
                    bins = np.linspace(min_val, max_val, 4)
                    df_processed[f'{var}_Category'] = pd.cut(df_processed[var], bins=bins, labels=labels, include_lowest=True)
        
        # Encontrar patrones en los datos de entrada
        category_cols = [f'{v}_Category' for v in variables_discretizadas if f'{v}_Category' in df_processed.columns]
        
        if category_cols:
            # Contar combinaciones en los datos de entrada
            input_combinations = df_processed[category_cols].value_counts().reset_index(name='Count')
            
            # Crear descripción de patrones
            df_processed['patron_climatico'] = 'Normal'
            
            # Asignar patrones basados en las combinaciones más frecuentes
            for idx, row in input_combinations.head(3).iterrows():
                pattern_desc = []
                for col in category_cols:
                    if col in row:
                        pattern_desc.append(f"{col.replace('_Category', '')}: {row[col]}")
                
                if pattern_desc:
                    pattern_name = f"Patrón {idx+1}: {', '.join(pattern_desc)}"
                    # Asignar este patrón a las filas que coincidan
                    mask = True
                    for col in category_cols:
                        if col in row:
                            mask = mask & (df_processed[col] == row[col])
                    
                    df_processed.loc[mask, 'patron_climatico'] = pattern_name
        
        if 'Location' not in df_processed.columns:
            df_processed['Location'] = ['Lugar ' + str(i) for i in range(len(df_processed))]
        
        # Preparar respuesta con información de patrones
        response_data = []
        for idx, row in df_processed.iterrows():
            response_data.append({
                'Location': row['Location'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'patron_climatico': row['patron_climatico'],
                'MaxTemp': row.get('MaxTemp', 'N/A'),
                'Humidity3pm': row.get('Humidity3pm', 'N/A'),
                'WindGustSpeed': row.get('WindGustSpeed', 'N/A'),
                'Sunshine': row.get('Sunshine', 'N/A')
            })
        
        return jsonify(response_data)
    else:
        raise ValueError("Se requieren columnas Latitud y Longitud para el análisis de asociación")

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port)
