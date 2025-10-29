"""
Configuración de la aplicación Streamlit para predicción de consumo de sustancias.
"""

import os

# Configuración de rutas
MODEL_PATH = "models/mejor_modelo_final.pkl"
PREPROCESSOR_PATH = "data/processed/preprocessor.joblib"
FEATURE_NAMES_PATH = "data/processed/feature_names.pkl"
METADATA_PATH = "models/metadatos_modelo_final.json"

# Configuración de la aplicación
APP_TITLE = "Predictor de Consumo de Sustancias Psicoactivas"
APP_SUBTITLE = "Análisis de Factores de Riesgo Sociales y Ambientales"
PAGE_ICON = "🧠"
LAYOUT = "wide"

# Mapeo de variables a nombres descriptivos
VARIABLE_DESCRIPTIONS = {
    'G_01_1.0': 'Familiares Consumen Sustancias',
    'G_02_1.0': 'Amigos Consumen Sustancias', 
    'G_03_1.0': 'Curiosidad por Probar',
    'G_04_1.0': 'Disposición a Consumir',
    'G_05_1.0': 'Oportunidad de Consumo',
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

# Categorías de variables
CATEGORICAL_VARS = ['G_01_1.0', 'G_02_1.0', 'G_03_1.0', 'G_04_1.0', 'G_05_1.0']
NUMERICAL_VARS = ['G_01_A', 'G_02_A', 'G_06_A', 'G_06_B', 'G_06_C', 'G_06_D', 'G_07', 'G_08_A', 'G_08_B']

# Opciones para variables categóricas
BINARY_OPTIONS = ['No', 'Sí']

# Rangos para variables numéricas
VARIABLE_RANGES = {
    'G_01_A': (0, 10, 0),  # (min, max, default)
    'G_02_A': (0, 20, 0),
    'G_06_A': (1, 5, 1),
    'G_06_B': (1, 5, 1),
    'G_06_C': (1, 5, 1),
    'G_06_D': (1, 5, 1),
    'G_07': (0, 50, 0),
    'G_08_A': (0, 20, 0),
    'G_08_B': (0, 10, 0)
}

# Colores para visualizaciones
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'warning': '#ff9800',
    'danger': '#d62728',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40'
}

# Umbrales de riesgo
RISK_THRESHOLDS = {
    'bajo': 0.3,
    'medio': 0.6,
    'alto': 1.0
}

# Configuración de SHAP
SHAP_CONFIG = {
    'max_display': 10,
    'plot_size': (10, 6)
}

# Mensajes de la aplicación
MESSAGES = {
    'loading': 'Cargando modelo...',
    'prediction_success': 'Predicción realizada exitosamente',
    'upload_success': 'Archivo cargado correctamente',
    'error_model': 'Error al cargar el modelo',
    'error_prediction': 'Error en la predicción',
    'no_data': 'No hay datos para mostrar'
}

# Configuración de Streamlit
STREAMLIT_CONFIG = {
    'page_title': APP_TITLE,
    'page_icon': PAGE_ICON,
    'layout': LAYOUT,
    'initial_sidebar_state': 'expanded'
}