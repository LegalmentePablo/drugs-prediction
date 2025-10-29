"""
Funciones auxiliares para la aplicación de predicción de consumo de sustancias.
"""

import pandas as pd
import numpy as np
import joblib
import pickle
import json
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
from config import *

@st.cache_resource
def load_model_and_preprocessor():
    """
    Carga el modelo entrenado y el preprocesador.
    """
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        
        with open(FEATURE_NAMES_PATH, 'rb') as f:
            feature_names = pickle.load(f)
            
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
            
        return model, preprocessor, feature_names, metadata
    except Exception as e:
        st.error(f"Error cargando modelo: {str(e)}")
        return None, None, None, None

def prepare_input_data(user_inputs):
    """
    Prepara los datos de entrada del usuario para la predicción.
    """
    # Crear DataFrame con los inputs del usuario usando nombres originales
    data = {}
    
    # Mapeo de variables categóricas (convertir de nombres con .0 a nombres originales)
    categorical_mapping = {
        'G_01_1.0': 'G_01',
        'G_02_1.0': 'G_02', 
        'G_03_1.0': 'G_03',
        'G_04_1.0': 'G_04',
        'G_05_1.0': 'G_05'
    }
    
    # Variables categóricas (convertir Sí/No a 1/0)
    for display_var, original_var in categorical_mapping.items():
        data[original_var] = 1.0 if user_inputs.get(display_var, 'No') == 'Sí' else 0.0
    
    # Variables numéricas (ya tienen los nombres correctos)
    for var in NUMERICAL_VARS:
        data[var] = float(user_inputs.get(var, VARIABLE_RANGES[var][2]))
    
    # Crear DataFrame
    df = pd.DataFrame([data])
    
    return df

def predict_consumption(user_inputs, model, preprocessor):
    """
    Realiza la predicción de consumo basada en los inputs del usuario.
    """
    try:
        # Preparar datos
        input_df = prepare_input_data(user_inputs)
        
        # Aplicar preprocesamiento
        X_processed = preprocessor.transform(input_df)
        
        # Realizar predicción
        prediction_proba = model.predict_proba(X_processed)[0]
        prediction = model.predict(X_processed)[0]
        
        return {
            'probability': prediction_proba[1],  # Probabilidad de consumo
            'prediction': prediction,
            'risk_level': get_risk_level(prediction_proba[1])
        }
    except Exception as e:
        st.error(f"Error en predicción: {str(e)}")
        return None

def get_risk_level(probability):
    """
    Determina el nivel de riesgo basado en la probabilidad.
    """
    if probability < RISK_THRESHOLDS['bajo']:
        return 'Bajo'
    elif probability < RISK_THRESHOLDS['medio']:
        return 'Medio'
    else:
        return 'Alto'

def get_risk_color(risk_level):
    """
    Retorna el color asociado al nivel de riesgo.
    """
    color_map = {
        'Bajo': COLORS['success'],
        'Medio': COLORS['warning'],
        'Alto': COLORS['danger']
    }
    return color_map.get(risk_level, COLORS['info'])

def create_probability_gauge(probability, risk_level):
    """
    Crea un gráfico de gauge para mostrar la probabilidad de consumo.
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Probabilidad de Consumo (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': get_risk_color(risk_level)},
            'steps': [
                {'range': [0, 30], 'color': COLORS['success']},
                {'range': [30, 60], 'color': COLORS['warning']},
                {'range': [60, 100], 'color': COLORS['danger']}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig

def create_feature_importance_plot(user_inputs, model, preprocessor, feature_names):
    """
    Crea un gráfico de importancia de características usando SHAP.
    """
    try:
        # Preparar datos
        input_df = prepare_input_data(user_inputs)
        X_processed = preprocessor.transform(input_df)
        
        # Asegurar que X_processed sea un array 2D
        if hasattr(X_processed, 'toarray'):
            X_processed = X_processed.toarray()
        
        # Crear explainer SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_processed)
        
        # Manejar diferentes formatos de shap_values
        if isinstance(shap_values, list):
            # Formato lista: [clase_0, clase_1]
            shap_values_single = shap_values[1][0]  # Clase positiva, primera muestra
        elif len(shap_values.shape) == 3:
            # Formato 3D: (muestras, características, clases)
            shap_values_single = shap_values[0, :, 1]  # Primera muestra, clase positiva
        elif len(shap_values.shape) == 2:
            # Formato 2D: (muestras, características)
            shap_values_single = shap_values[0]  # Primera muestra
        else:
            # Formato 1D: (características)
            shap_values_single = shap_values
        
        # Mapeo de nombres originales a nombres descriptivos
        original_to_display = {
            'G_01': 'Familiares Consumen Sustancias',
            'G_02': 'Amigos Consumen Sustancias',
            'G_03': 'Curiosidad por Probar',
            'G_04': 'Disposición a Consumir',
            'G_05': 'Oportunidad de Consumo',
            'G_01_A': 'Cantidad de Familiares que Consumen',
            'G_02_A': 'Cantidad de Amigos que Consumen',
            'G_06_A': 'Facilidad Acceso Marihuana (1-5)',
            'G_06_B': 'Facilidad Acceso Cocaína (1-5)',
            'G_06_C': 'Facilidad Acceso Basuco (1-5)',
            'G_06_D': 'Facilidad Acceso Éxtasis (1-5)',
            'G_07': 'Ofertas Recibidas (Total)',
            'G_08_A': 'Ofertas de Marihuana',
            'G_08_B': 'Ofertas de Cocaína'
        }
        
        # Obtener los nombres de características del preprocessor
        input_feature_names = preprocessor.feature_names_in_
        
        # Obtener valores de características para la primera muestra
        X_values_single = X_processed[0] if len(X_processed.shape) > 1 else X_processed
        
        # Verificar que las dimensiones coincidan
        if len(shap_values_single) != len(input_feature_names):
            st.error(f"Dimensiones no coinciden: SHAP values {len(shap_values_single)}, features {len(input_feature_names)}")
            return None
        
        # Crear DataFrame para el gráfico
        feature_importance = pd.DataFrame({
            'feature': [original_to_display.get(feat, feat) for feat in input_feature_names],
            'importance': shap_values_single,
            'value': X_values_single
        })
        
        # Ordenar por importancia absoluta
        feature_importance['abs_importance'] = abs(feature_importance['importance'])
        feature_importance = feature_importance.sort_values('abs_importance', ascending=True)
        
        # Crear gráfico de barras
        fig = px.bar(
            feature_importance.tail(10),  # Top 10 características
            x='importance',
            y='feature',
            orientation='h',
            color='importance',
            color_continuous_scale='RdYlBu_r',
            title='Importancia de Características (SHAP Values)',
            labels={'importance': 'Contribución a la Predicción', 'feature': 'Característica'}
        )
        
        fig.update_layout(height=500, showlegend=False)
        return fig
        
    except Exception as e:
        st.error(f"Error creando explicación SHAP: {str(e)}")
        return None

def create_comparison_chart(user_probability):
    """
    Crea un gráfico comparativo con la población general.
    """
    # Datos simulados de población general (basados en prevalencia del 8%)
    population_data = {
        'Categoría': ['Usuario Actual', 'Población General'],
        'Probabilidad': [user_probability * 100, 8.0],
        'Color': [get_risk_color(get_risk_level(user_probability)), COLORS['info']]
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=population_data['Categoría'],
            y=population_data['Probabilidad'],
            marker_color=population_data['Color'],
            text=[f"{p:.1f}%" for p in population_data['Probabilidad']],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Comparación con Población General',
        yaxis_title='Probabilidad de Consumo (%)',
        showlegend=False,
        height=400
    )
    
    return fig

def process_batch_predictions(uploaded_file, model, preprocessor, feature_names):
    """
    Procesa predicciones en lote desde un archivo CSV.
    """
    try:
        # Leer archivo CSV
        df = pd.read_csv(uploaded_file)
        
        # Nombres de características esperados por el preprocessor
        expected_features = preprocessor.feature_names_in_
        
        # Verificar que las columnas necesarias estén presentes
        missing_cols = set(expected_features) - set(df.columns)
        if missing_cols:
            st.error(f"Columnas faltantes en el archivo: {missing_cols}")
            return None
        
        # Seleccionar solo las columnas necesarias
        df_features = df[expected_features]
        
        # Aplicar preprocesamiento
        X_processed = preprocessor.transform(df_features)
        
        # Realizar predicciones
        predictions = model.predict(X_processed)
        probabilities = model.predict_proba(X_processed)[:, 1]
        
        # Agregar resultados al DataFrame original
        df['Prediccion'] = predictions
        df['Probabilidad'] = probabilities
        df['Nivel_Riesgo'] = [get_risk_level(p) for p in probabilities]
        
        return df
        
    except Exception as e:
        st.error(f"Error procesando archivo: {str(e)}")
        return None

def create_batch_summary_charts(df_results):
    """
    Crea gráficos resumen para predicciones en lote.
    """
    # Distribución de niveles de riesgo
    risk_counts = df_results['Nivel_Riesgo'].value_counts()
    
    fig1 = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title='Distribución de Niveles de Riesgo',
        color_discrete_map={
            'Bajo': COLORS['success'],
            'Medio': COLORS['warning'],
            'Alto': COLORS['danger']
        }
    )
    
    # Histograma de probabilidades
    fig2 = px.histogram(
        df_results,
        x='Probabilidad',
        nbins=20,
        title='Distribución de Probabilidades de Consumo',
        labels={'Probabilidad': 'Probabilidad de Consumo', 'count': 'Frecuencia'}
    )
    
    return fig1, fig2

def generate_recommendations(risk_level, probability, user_inputs):
    """
    Genera recomendaciones personalizadas basadas en el nivel de riesgo.
    """
    recommendations = []
    
    if risk_level == 'Alto':
        recommendations.extend([
            "🚨 **Riesgo Alto Detectado**: Se recomienda buscar apoyo profesional inmediatamente.",
            "📞 **Líneas de Ayuda**: Contactar servicios de salud mental especializados.",
            "👥 **Apoyo Social**: Buscar grupos de apoyo y redes de contención familiar."
        ])
    elif risk_level == 'Medio':
        recommendations.extend([
            "⚠️ **Riesgo Moderado**: Importante implementar estrategias de prevención.",
            "🧠 **Educación**: Informarse sobre los riesgos del consumo de sustancias.",
            "🏃‍♂️ **Actividades Alternativas**: Buscar actividades recreativas y deportivas."
        ])
    else:
        recommendations.extend([
            "✅ **Riesgo Bajo**: Mantener factores protectores actuales.",
            "🛡️ **Prevención**: Continuar con hábitos saludables y entorno positivo.",
            "📚 **Información**: Mantenerse informado sobre prevención de adicciones."
        ])
    
    # Recomendaciones específicas basadas en factores de riesgo
    if user_inputs.get('G_01_1.0') == 'Sí':
        recommendations.append("👨‍👩‍👧‍👦 **Entorno Familiar**: Considerar terapia familiar o consejería.")
    
    if user_inputs.get('G_02_1.0') == 'Sí':
        recommendations.append("👫 **Círculo Social**: Evaluar influencias del grupo de amigos.")
    
    if user_inputs.get('G_03_1.0') == 'Sí' or user_inputs.get('G_04_1.0') == 'Sí':
        recommendations.append("🎯 **Manejo de Impulsos**: Técnicas de autocontrol y mindfulness.")
    
    return recommendations

def create_scenario_simulator():
    """
    Crea una interfaz para simular diferentes escenarios.
    """
    st.subheader("🎮 Simulador de Escenarios")
    
    scenarios = {
        "Escenario Base": {
            'G_01_1.0': 'No', 'G_02_1.0': 'No', 'G_03_1.0': 'No',
            'G_04_1.0': 'No', 'G_05_1.0': 'No', 'G_01_A': 0,
            'G_02_A': 0, 'G_06_A': 1, 'G_06_B': 1, 'G_06_C': 1,
            'G_06_D': 1, 'G_07': 0, 'G_08_A': 0, 'G_08_B': 0
        },
        "Entorno Familiar de Riesgo": {
            'G_01_1.0': 'Sí', 'G_02_1.0': 'No', 'G_03_1.0': 'No',
            'G_04_1.0': 'No', 'G_05_1.0': 'No', 'G_01_A': 2,
            'G_02_A': 0, 'G_06_A': 3, 'G_06_B': 2, 'G_06_C': 2,
            'G_06_D': 2, 'G_07': 1, 'G_08_A': 1, 'G_08_B': 0
        },
        "Presión Social Alta": {
            'G_01_1.0': 'No', 'G_02_1.0': 'Sí', 'G_03_1.0': 'Sí',
            'G_04_1.0': 'No', 'G_05_1.0': 'Sí', 'G_01_A': 0,
            'G_02_A': 5, 'G_06_A': 4, 'G_06_B': 3, 'G_06_C': 2,
            'G_06_D': 3, 'G_07': 3, 'G_08_A': 2, 'G_08_B': 1
        }
    }
    
    return scenarios