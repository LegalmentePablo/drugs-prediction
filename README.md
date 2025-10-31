# Predicción de Consumo de Sustancias Psicoactivas
## Análisis de Factores de Riesgo Sociales y Ambientales

### Descripción del Proyecto
Este proyecto utiliza la Encuesta Nacional de Consumo de Sustancias Psicoactivas (ENCSPA) 2019 del DANE para predecir patrones de consumo de drogas en Colombia mediante técnicas de Machine Learning.

### Objetivos
- **Principal**: Predecir consumo de marihuana basado en entorno social y accesibilidad
- **Secundarios**: Clasificar nivel de riesgo y identificar patrones de transición entre curiosidad y consumo

### Estructura del Proyecto
```
drugs-prediction/
├── .gitignore              # Exclusión de archivos pesados (.pkl)
├── data/                   # Datos del proyecto
│   ├── g_capitulos.csv     # Dataset ENCSPA 2019
│   ├── variables.pdf       # PDF con toda la información sobre las variables
│   └── processed/          # Datos procesados y preprocesador
│       ├── preprocessor.joblib
│       └── feature_names.pkl
├── models/                 # Modelos entrenados y metadatos
│   ├── mejor_modelo_final.pkl          # Modelo RF final (excluido por .gitignore, debido a su tamaño no permitido en github)
│   ├── metadatos_modelo_final.json     # Métricas y configuración del modelo
│   └── metadatos_randomforest.json     # Metadatos adicionales
├── notebooks/              # Análisis completo paso a paso
│   ├── 01_EDA_ENCSPA_2019.ipynb
│   ├── 02_preprocesamiento_datos.ipynb
│   ├── 03_encoding_transformaciones.ipynb
│   ├── 04_modelos_avanzados.ipynb
│   ├── 05_optimizacion_modelo_final.ipynb
│   └── 06_interpretabilidad_modelos.ipynb    # NUEVO: Análisis SHAP
├── app.py                  # Aplicación web Streamlit
├── utils.py                # Funciones auxiliares
├── config.py               # Configuración de la aplicación
├── requirements.txt        # Dependencias del proyecto
└── README.md              # Documentación del proyecto
```

### Fuente de los Datos

**Dataset**: Encuesta Nacional de Consumo de Sustancias Psicoactivas (ENCSPA) 2019
- **Institución**: Departamento Administrativo Nacional de Estadística (DANE) - Colombia
- **Año**: 2019
- **Población**: Personas de 12 a 65 años en territorio nacional colombiano
- **Tamaño de muestra**: 49,756 observaciones
- **Variables**: 98 variables relacionadas con consumo, entorno social y factores de riesgo
- **Acceso**: Datos públicos disponibles en portal DANE
- **Propósito original**: Caracterizar patrones de consumo de sustancias psicoactivas en Colombia
- **Portal de datos**: [DANE - Microdatos](https://microdatos.dane.gov.co/index.php/catalog/680)
- **Diccionario de variables**: [Capítulo G - Variables de consumo](https://microdatos.dane.gov.co/index.php/catalog/680/data-dictionary/F17?file_name=g_capitulos)
- **Documentación completa**: Incluye metodología, cuestionarios y definiciones de variables

**Relevancia para el proyecto**:
- Variables de entorno social bien documentadas
- Múltiples variables objetivo para análisis comparativo
- Datos representativos a nivel nacional
- Variables predictoras validadas sociológicamente

### Variables Objetivo Identificadas
1. **Marihuana (G_11_F)**: 8.0% prevalencia - Variable principal
2. **Cocaína (G_11_G)**: 1.96% prevalencia - Variable intermedia  
3. **Basuco (G_11_H)**: 0.62% prevalencia - Variable alto riesgo
4. **Jeringas (G_13)**: 0.014% prevalencia - Análisis descriptivo

### Variables Predictoras Principales (15 Variables Total)
- **Entorno Social** (Variables categóricas): 
  - Familiares consumidores (G_01), Amigos consumidores (G_02)
- **Actitudes** (Variables categóricas): 
  - Curiosidad (G_03), Disposición (G_04), Oportunidad (G_05)
- **Accesibilidad** (Variables numéricas): 
  - Facilidad de acceso: Marihuana (G_06_A), Cocaína (G_06_B), Basuco (G_06_C), Éxtasis (G_06_D)
- **Exposición** (Variables numéricas): 
  - Ofertas recibidas (G_07), Ofertas marihuana (G_08_A), Ofertas cocaína (G_08_B)
- **Cantidades** (Variables numéricas): 
  - Cantidad familiares (G_01_A), Cantidad amigos (G_02_A)

### Flujo de Desarrollo Completo
1. **`01_EDA_ENCSPA_2019.ipynb`**: Análisis exploratorio y definición de variables
2. **`02_preprocesamiento_datos.ipynb`**: Limpieza y preparación de datos
3. **`03_encoding_transformaciones.ipynb`**: Encoding categóricas + transformaciones
4. **`04_modelos_avanzados.ipynb`**: Modelos ML con 5-fold CV + análisis comparativo
5. **`05_optimizacion_modelo_final.ipynb`**: Optuna tuning + ensemble methods + selección final
6. **`06_interpretabilidad_modelos.ipynb`**: **NUEVO**
   - **SHAP Analysis**: TreeExplainer para explicabilidad del modelo Random Forest
   - **Feature Importance**: Análisis nativo + Permutation Importance 
   - **Visualizaciones Plotly**: Gráficos interactivos profesionales
   - **Casos específicos**: Alto riesgo, falsos positivos/negativos
   - **Insights para políticas**: Variables más predictivas identificadas

### Rendimiento del Modelo Final
- **Algoritmo seleccionado**: Random Forest (mejor F1-Score tras comparación)
- **Métricas en test set**:
  - **F1-Score**: 0.662
  - **ROC-AUC**: 0.918  
  - **Accuracy**: 0.875
  - **Precision**: 0.585
  - **Recall**: 0.762
- **Metodología**: 5-fold CV + Optuna optimization (220 trials) + SMOTE balancing
- **Dataset balanceado**: 31,442 muestras entrenamiento (50% balance), 4,686 test (16.1% positivos)

### Hallazgos Principales del Análisis
#### Variables más Predictivas (Feature Importance)
1. **Curiosidad por Probar** (43.1%) - Factor psicológico dominante
2. **Tuvo Oportunidad de Probar** (15.4%) - Factor situacional clave
3. **Disposición a Consumir** (11.7%) - Factor actitudinal relevante

#### Categorización de Factores de Riesgo
- **Actitudes Personales**: 70.2% de importancia total
- **Accesibilidad**: 12.6% de importancia
- **Exposición**: 12.5% de importancia  
- **Entorno Social**: 5.8% de importancia

**Patrón clave identificado**: Las actitudes individuales superan 12x al entorno social como predictor, sugiriendo que el consumo es más una decisión personal influenciada por curiosidad que por presión social.

#### Umbrales de Riesgo para Intervención
- **Alto riesgo**: Probabilidad > 0.833 (Percentil 90)
- **Riesgo crítico**: Probabilidad > 0.917 (Percentil 95)
- **Recomendación**: Monitorear casos con probabilidad > 0.7 para intervención temprana

### ⚠️ Nota sobre Modelos Entrenados
Los archivos `.pkl` de modelos entrenados (>100MB) están excluidos del repositorio via `.gitignore` por las limitaciones de GitHub. Para reproducir:
1. Ejecutar notebooks 04-05 para entrenar modelos desde cero
2. Los metadatos completos están disponibles en `models/metadatos_modelo_final.json`
3. El preprocesador está guardado en `data/processed/preprocessor.joblib`

### Tecnologías Utilizadas
- **Análisis y EDA**: Python, Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost, Random Forest
- **Interpretabilidad**: SHAP
- **Visualización Avanzada**: Plotly, Matplotlib, Seaborn
- **Notebooks Interactivos**: Jupyter Lab/Notebook
- **Aplicación Web**: Streamlit
- **Control de versiones**: Git, GitHub con flujo de trabajo colaborativo

### Equipo de Desarrollo
- **Ramas de desarrollo**: 
  - `dev-pablo`
  - `dev-amaury`
- **Asignaciones de Commits**:
  - Commit 2: Pablo
  - Commit 3: Encoding y transformaciones (Amaury)
  - Commit 4: Modelos avanzados ML (Pablo)
  - Commit 5: Optimización final (Amaury)
  - Commit 6: Interpretabilidad (Pablo)
  - Commit 7: App Streamlit (Amaury)
  - Commit 8: Validación externa (Pablo)
  - Commit 9: Documentación final (Amaury)
- **Metodología**: 
  - Commits incrementales con notebooks modulares
  - Pull Requests para integración a `main`
  - Revisión de código y sincronización de variables

## Aplicación Web Streamlit

### Características de la Aplicación
La aplicación web desarrollada en Streamlit proporciona una interfaz interactiva para:

1. **Predicción Individual**: 
   - Formulario intuitivo para ingresar factores de riesgo
   - Predicción en tiempo real del riesgo de consumo
   - Explicaciones interpretables con SHAP

2. **Dashboard General**:
   - Visualizaciones interactivas del dataset
   - Estadísticas descriptivas por variables
   - Análisis de correlaciones

3. **Predicción por Lotes**:
   - Carga de archivos CSV para múltiples predicciones
   - Descarga de resultados procesados
   - Plantilla CSV para facilitar el formato

4. **Simulador de Escenarios**:
   - Análisis de sensibilidad de variables
   - Recomendaciones personalizadas
   - Comparación de diferentes perfiles de riesgo

5. **Información del Modelo**:
   - Métricas de rendimiento
   - Descripción de variables
   - Metodología utilizada

### Instalación y Ejecución

#### Requisitos Previos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

#### Instalación de Dependencias
```bash
# Clonar el repositorio
git clone https://github.com/LegalmentePablo/am-pa-drugs-prediction.git
cd drugs-prediction

# Instalar dependencias
pip install -r requirements.txt
```

#### Ejecución de la Aplicación
```bash
# Ejecutar la aplicación Streamlit
streamlit run app.py
```

La aplicación estará disponible en:
- **URL Local**: http://localhost:8501
- **URL de Red**: http://[tu-ip]:8501

#### Comandos Adicionales
```bash
# Ejecutar en un puerto específico
streamlit run app.py --server.port 8502

# Ejecutar con configuración personalizada
streamlit run app.py --server.headless true

# Ver ayuda de Streamlit
streamlit --help
```

### Estructura de Archivos de la Aplicación
- **`app.py`**: Archivo principal de la aplicación Streamlit
- **`utils.py`**: Funciones auxiliares para procesamiento y visualización
- **`config.py`**: Configuración y constantes de la aplicación
- **`requirements.txt`**: Lista de dependencias necesarias

### Uso de la Aplicación
1. **Acceder a la aplicación** en http://localhost:8501
2. **Seleccionar una pestaña** según el tipo de análisis deseado
3. **Ingresar datos** usando los controles interactivos
4. **Visualizar resultados** y explicaciones en tiempo real
5. **Descargar resultados** cuando sea aplicable

---
*Proyecto desarrollado como parte del curso de Ciencia de Datos e Inteligencia Artificial*