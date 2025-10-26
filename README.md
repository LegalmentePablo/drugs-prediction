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
├── data/                    # Datos del proyecto
│   └── g_capitulos.csv     # Dataset ENCSPA 2019
|   └── variables.pdf       # PDF con toda la información sobre las variables
├── notebooks/              # Análisis exploratorio
│   ├── 01_EDA_ENCSPA_2019.ipynb        # Análisis Exploratorio de Datos
│   └── 02_preprocesamiento_datos.ipynb  # Preprocesamiento y Limpieza
├── docs/                   # Documentación
└── tests/                  # Pruebas unitarias
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

### Tecnologías Utilizadas
- **Análisis y EDA**: Python, Pandas, NumPy
- **Visualización Avanzada**: Plotly
- **Notebooks Interactivos**: Jupyter Lab/Notebook
- **Aplicación Web** (futuro): Streamlit
- **Control de versiones**: Git, GitHub con flujo de trabajo colaborativo

### 👥 Equipo de Desarrollo
- **Ramas de desarrollo**: 
  - `dev-pablo`
  - `dev-amaury`
- **Metodología**: 
  - Commits incrementales con notebooks modulares
  - Pull Requests para integración a `main`
  - Revisión de código y sincronización de variables

---
*Proyecto desarrollado como parte del curso de Ciencia de Datos e Inteligencia Artificial*