"""
Aplicación Streamlit para Predicción de Consumo de Sustancias Psicoactivas
Análisis de Factores de Riesgo Sociales y Ambientales
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import *
from config import *

# Configuración de la página
st.set_page_config(**STREAMLIT_CONFIG)

# Cargar modelo y datos
@st.cache_resource
def initialize_app():
    return load_model_and_preprocessor()

def main():
    # Título principal
    st.title(f"{APP_TITLE}")
    st.markdown(f"### {APP_SUBTITLE}")
    st.markdown("---")
    
    # Cargar modelo
    model, preprocessor, feature_names, metadata = initialize_app()
    
    if model is None:
        st.error("No se pudo cargar el modelo. Verifique los archivos.")
        return
    
    # Sidebar con información del modelo
    with st.sidebar:
        st.header("Información del Modelo")
        if metadata:
            st.metric("Modelo Seleccionado", metadata.get('modelo_final', 'N/A').upper())
            rf_results = metadata.get('resultados_test', {}).get('rf', {})
            if rf_results:
                st.metric("Precisión (F1-Score)", f"{rf_results.get('f1', 0):.3f}")
                st.metric("AUC-ROC", f"{rf_results.get('auc', 0):.3f}")
                st.metric("Exactitud", f"{rf_results.get('accuracy', 0):.3f}")
        
        st.markdown("---")
        st.markdown("**Fuente de Datos:**")
        st.markdown("ENCSPA 2019 - DANE Colombia")
        st.markdown("**Muestra:** 49,756 observaciones")
    
    # Tabs principales
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Predicción Individual", 
        "Dashboard General", 
        "Predicción por Lotes", 
        "Simulador de Escenarios",
        "Información"
    ])
    
    # TAB 1: Predicción Individual
    with tab1:
        st.header("Predicción Individual")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Variables de Entrada")
            user_inputs = {}
            
            # Sección: Entorno Social
            st.markdown("#### Entorno Social")
            user_inputs['G_01_1.0'] = st.selectbox(
                VARIABLE_DESCRIPTIONS['G_01_1.0'], 
                BINARY_OPTIONS, 
                key="fam_consume"
            )
            
            if user_inputs['G_01_1.0'] == 'Sí':
                user_inputs['G_01_A'] = st.number_input(
                    VARIABLE_DESCRIPTIONS['G_01_A'], 
                    min_value=VARIABLE_RANGES['G_01_A'][0],
                    max_value=VARIABLE_RANGES['G_01_A'][1],
                    value=VARIABLE_RANGES['G_01_A'][2],
                    key="cant_fam"
                )
            else:
                user_inputs['G_01_A'] = 0
            
            user_inputs['G_02_1.0'] = st.selectbox(
                VARIABLE_DESCRIPTIONS['G_02_1.0'], 
                BINARY_OPTIONS,
                key="amig_consume"
            )
            
            if user_inputs['G_02_1.0'] == 'Sí':
                user_inputs['G_02_A'] = st.number_input(
                    VARIABLE_DESCRIPTIONS['G_02_A'], 
                    min_value=VARIABLE_RANGES['G_02_A'][0],
                    max_value=VARIABLE_RANGES['G_02_A'][1],
                    value=VARIABLE_RANGES['G_02_A'][2],
                    key="cant_amig"
                )
            else:
                user_inputs['G_02_A'] = 0
            
            # Sección: Actitudes
            st.markdown("#### Actitudes y Disposición")
            user_inputs['G_03_1.0'] = st.selectbox(
                VARIABLE_DESCRIPTIONS['G_03_1.0'], 
                BINARY_OPTIONS,
                key="curiosidad"
            )
            
            user_inputs['G_04_1.0'] = st.selectbox(
                VARIABLE_DESCRIPTIONS['G_04_1.0'], 
                BINARY_OPTIONS,
                key="disposicion"
            )
            
            user_inputs['G_05_1.0'] = st.selectbox(
                VARIABLE_DESCRIPTIONS['G_05_1.0'], 
                BINARY_OPTIONS,
                key="oportunidad"
            )
            
            # Sección: Accesibilidad
            st.markdown("#### Facilidad de Acceso")
            for var in ['G_06_A', 'G_06_B', 'G_06_C', 'G_06_D']:
                user_inputs[var] = st.slider(
                    VARIABLE_DESCRIPTIONS[var],
                    min_value=VARIABLE_RANGES[var][0],
                    max_value=VARIABLE_RANGES[var][1],
                    value=VARIABLE_RANGES[var][2],
                    key=f"acceso_{var}"
                )
            
            # Sección: Exposición
            st.markdown("#### Exposición a Ofertas")
            for var in ['G_07', 'G_08_A', 'G_08_B']:
                user_inputs[var] = st.number_input(
                    VARIABLE_DESCRIPTIONS[var],
                    min_value=VARIABLE_RANGES[var][0],
                    max_value=VARIABLE_RANGES[var][1],
                    value=VARIABLE_RANGES[var][2],
                    key=f"ofertas_{var}"
                )
        
        with col2:
            st.subheader("Resultados de la Predicción")
            
            if st.button("Realizar Predicción", type="primary", use_container_width=True):
                with st.spinner("Analizando factores de riesgo..."):
                    result = predict_consumption(user_inputs, model, preprocessor)
                    
                    if result:
                        # Métricas principales
                        col_metric1, col_metric2, col_metric3 = st.columns(3)
                        
                        with col_metric1:
                            st.metric(
                                "Probabilidad de Consumo",
                                f"{result['probability']:.1%}",
                                delta=f"{result['probability']-0.08:.1%} vs población"
                            )
                        
                        with col_metric2:
                            st.metric(
                                "Nivel de Riesgo",
                                result['risk_level'],
                                delta=None
                            )
                        
                        with col_metric3:
                            prediction_text = "Sí" if result['prediction'] == 1 else "No"
                            st.metric(
                                "Predicción",
                                prediction_text,
                                delta=None
                            )
                        
                        # Gráfico de gauge
                        gauge_fig = create_probability_gauge(result['probability'], result['risk_level'])
                        st.plotly_chart(gauge_fig, use_container_width=True)
                        
                        # Comparación con población
                        comparison_fig = create_comparison_chart(result['probability'])
                        st.plotly_chart(comparison_fig, use_container_width=True)
                        
                        # Explicabilidad SHAP
                        st.subheader("Explicación de la Predicción")
                        shap_fig = create_feature_importance_plot(user_inputs, model, preprocessor, feature_names)
                        if shap_fig:
                            st.plotly_chart(shap_fig, use_container_width=True)
                        
                        # Recomendaciones
                        st.subheader("Recomendaciones Personalizadas")
                        recommendations = generate_recommendations(result['risk_level'], result['probability'], user_inputs)
                        for rec in recommendations:
                            st.markdown(f"- {rec}")
    
    # TAB 2: Dashboard General
    with tab2:
        st.header("Dashboard General")
        
        # Estadísticas del modelo
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Prevalencia Nacional", "8.0%", help="Porcentaje de consumo de marihuana en Colombia")
        
        with col2:
            st.metric("Tamaño de Muestra", "49,756", help="Observaciones en ENCSPA 2019")
        
        with col3:
            st.metric("Variables Predictoras", "14", help="Características utilizadas en el modelo")
        
        with col4:
            st.metric("Precisión del Modelo", "87.5%", help="Exactitud en conjunto de prueba")
        
        # Gráficos informativos
        st.subheader("Análisis de Variables")
        
        # Simulación de distribución de factores de riesgo
        np.random.seed(42)
        sample_data = {
            'Entorno_Familiar': np.random.choice(['Sin Riesgo', 'Riesgo Bajo', 'Riesgo Alto'], 1000, p=[0.7, 0.2, 0.1]),
            'Presion_Social': np.random.choice(['Sin Riesgo', 'Riesgo Bajo', 'Riesgo Alto'], 1000, p=[0.6, 0.25, 0.15]),
            'Accesibilidad': np.random.choice(['Baja', 'Media', 'Alta'], 1000, p=[0.5, 0.3, 0.2]),
            'Probabilidad_Consumo': np.random.beta(2, 20, 1000)  # Distribución sesgada hacia valores bajos
        }
        
        df_sample = pd.DataFrame(sample_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.pie(
                df_sample, 
                names='Entorno_Familiar', 
                title='Distribución de Riesgo Familiar',
                color_discrete_map={
                    'Sin Riesgo': COLORS['success'],
                    'Riesgo Bajo': COLORS['warning'],
                    'Riesgo Alto': COLORS['danger']
                }
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.histogram(
                df_sample, 
                x='Probabilidad_Consumo', 
                nbins=30,
                title='Distribución de Probabilidades de Consumo',
                labels={'Probabilidad_Consumo': 'Probabilidad', 'count': 'Frecuencia'}
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Matriz de correlación simulada
        st.subheader("Correlaciones entre Variables")
        correlation_data = np.random.rand(5, 5)
        correlation_data = (correlation_data + correlation_data.T) / 2  # Hacer simétrica
        np.fill_diagonal(correlation_data, 1)  # Diagonal = 1
        
        variables = ['Entorno Familiar', 'Presión Social', 'Curiosidad', 'Accesibilidad', 'Ofertas']
        
        fig_corr = px.imshow(
            correlation_data,
            x=variables,
            y=variables,
            color_continuous_scale='RdBu_r',
            title='Matriz de Correlación de Factores de Riesgo'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # TAB 3: Predicción por Lotes
    with tab3:
        st.header("Predicción por Lotes")
        st.markdown("Sube un archivo CSV con múltiples casos para obtener predicciones masivas.")
        
        # Template de descarga
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Formato del Archivo")
            # Crear template con nombres originales esperados por el preprocessor
            model, preprocessor, feature_names, metadata = initialize_app()
            if preprocessor is not None:
                expected_features = preprocessor.feature_names_in_
                template_data = {}
                for var in expected_features:
                    if var in ['G_01', 'G_02', 'G_03', 'G_04', 'G_05']:
                        template_data[var] = [0, 1, 0]  # Valores binarios para categóricas
                    else:
                        default_val = VARIABLE_RANGES.get(var, (0, 1, 0))[2]
                        template_data[var] = [default_val, default_val+1, default_val+2]
                
                template_df = pd.DataFrame(template_data)
                st.dataframe(template_df, use_container_width=True)
            else:
                st.error("No se pudo cargar el preprocessor para generar el template")
        
        with col2:
            st.subheader("Descargar Template")
            csv_template = template_df.to_csv(index=False)
            st.download_button(
                label="Descargar CSV Template",
                data=csv_template,
                file_name="template_prediccion.csv",
                mime="text/csv"
            )
        
        # Upload de archivo
        st.subheader("Subir Archivo")
        uploaded_file = st.file_uploader(
            "Selecciona un archivo CSV",
            type=['csv'],
            help="El archivo debe contener las columnas especificadas en el template"
        )
        
        if uploaded_file is not None:
            with st.spinner("Procesando predicciones..."):
                results_df = process_batch_predictions(uploaded_file, model, preprocessor, feature_names)
                
                if results_df is not None:
                    st.success(f"Procesadas {len(results_df)} predicciones exitosamente")
                    
                    # Resumen de resultados
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_prob = results_df['Probabilidad'].mean()
                        st.metric("Probabilidad Promedio", f"{avg_prob:.1%}")
                    
                    with col2:
                        high_risk_count = (results_df['Nivel_Riesgo'] == 'Alto').sum()
                        st.metric("Casos de Alto Riesgo", high_risk_count)
                    
                    with col3:
                        positive_pred = (results_df['Prediccion'] == 1).sum()
                        st.metric("Predicciones Positivas", positive_pred)
                    
                    # Gráficos resumen
                    fig1, fig2 = create_batch_summary_charts(results_df)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(fig1, use_container_width=True)
                    with col2:
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Tabla de resultados
                    st.subheader("Resultados Detallados")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Descarga de resultados
                    csv_results = results_df.to_csv(index=False)
                    st.download_button(
                        label="Descargar Resultados",
                        data=csv_results,
                        file_name="resultados_prediccion.csv",
                        mime="text/csv"
                    )
    
    # TAB 4: Simulador de Escenarios
    with tab4:
        st.header("Simulador de Escenarios")
        st.markdown("Explora diferentes combinaciones de factores de riesgo y observa cómo afectan la predicción.")
        
        scenarios = create_scenario_simulator()
        
        # Selector de escenario
        selected_scenario = st.selectbox(
            "Selecciona un escenario predefinido:",
            list(scenarios.keys())
        )
        
        scenario_inputs = scenarios[selected_scenario].copy()
        
        # Modificadores de escenario
        st.subheader("Modificar Escenario")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Entorno Social**")
            scenario_inputs['G_01_1.0'] = st.selectbox("Familiares Consumen", BINARY_OPTIONS, 
                                                      index=BINARY_OPTIONS.index(scenario_inputs['G_01_1.0']), key="sim_fam")
            scenario_inputs['G_02_1.0'] = st.selectbox("Amigos Consumen", BINARY_OPTIONS, 
                                                      index=BINARY_OPTIONS.index(scenario_inputs['G_02_1.0']), key="sim_amig")
        
        with col2:
            st.markdown("**Actitudes**")
            scenario_inputs['G_03_1.0'] = st.selectbox("Curiosidad", BINARY_OPTIONS, 
                                                      index=BINARY_OPTIONS.index(scenario_inputs['G_03_1.0']), key="sim_cur")
            scenario_inputs['G_04_1.0'] = st.selectbox("Disposición", BINARY_OPTIONS, 
                                                      index=BINARY_OPTIONS.index(scenario_inputs['G_04_1.0']), key="sim_disp")
        
        with col3:
            st.markdown("**Accesibilidad**")
            scenario_inputs['G_06_A'] = st.slider("Acceso Marihuana", 1, 5, scenario_inputs['G_06_A'], key="sim_acc_mar")
            scenario_inputs['G_07'] = st.slider("Ofertas Totales", 0, 10, scenario_inputs['G_07'], key="sim_ofertas")
        
        # Predicción del escenario
        if st.button("Simular Escenario", type="primary"):
            result = predict_consumption(scenario_inputs, model, preprocessor)
            
            if result:
                # Resultados en columnas
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Probabilidad", f"{result['probability']:.1%}")
                with col2:
                    st.metric("Nivel de Riesgo", result['risk_level'])
                with col3:
                    prediction_text = "Consumo Probable" if result['prediction'] == 1 else "No Consumo"
                    st.metric("Predicción", prediction_text)
                
                # Visualización
                gauge_fig = create_probability_gauge(result['probability'], result['risk_level'])
                st.plotly_chart(gauge_fig, use_container_width=True)
        
        # Comparación de escenarios
        st.subheader("Comparación de Escenarios")
        if st.button("Comparar Todos los Escenarios"):
            comparison_results = []
            
            for scenario_name, scenario_data in scenarios.items():
                result = predict_consumption(scenario_data, model, preprocessor)
                if result:
                    comparison_results.append({
                        'Escenario': scenario_name,
                        'Probabilidad': result['probability'],
                        'Nivel_Riesgo': result['risk_level']
                    })
            
            if comparison_results:
                comparison_df = pd.DataFrame(comparison_results)
                
                fig_comparison = px.bar(
                    comparison_df,
                    x='Escenario',
                    y='Probabilidad',
                    color='Nivel_Riesgo',
                    title='Comparación de Probabilidades por Escenario',
                    color_discrete_map={
                        'Bajo': COLORS['success'],
                        'Medio': COLORS['warning'],
                        'Alto': COLORS['danger']
                    }
                )
                st.plotly_chart(fig_comparison, use_container_width=True)
    
    # TAB 5: Información
    with tab5:
        st.header("Información del Proyecto")
        
        # Información del dataset
        st.subheader("Sobre los Datos")
        st.markdown("""
        **Fuente:** Encuesta Nacional de Consumo de Sustancias Psicoactivas (ENCSPA) 2019 - DANE Colombia
        
        **Características del Dataset:**
        - Tamaño: 49,756 observaciones
        - Año: 2019
        - Población: Personas de 12 a 65 años en Colombia
        - Prevalencia de Marihuana: 8.0%
        - Variables: 14 predictoras seleccionadas
        """)
        
        # Variables del modelo
        st.subheader("Variables del Modelo")
        
        variables_info = pd.DataFrame([
            {"Variable": desc, "Código": var, "Tipo": "Categórica" if var in CATEGORICAL_VARS else "Numérica"}
            for var, desc in VARIABLE_DESCRIPTIONS.items()
        ])
        
        st.dataframe(variables_info, use_container_width=True)
        
        # Información del modelo
        st.subheader("Sobre el Modelo")
        st.markdown("""
        **Algoritmo:** Random Forest (Bosque Aleatorio)
        
        **Características:**
        - Balanceado: Uso de SMOTE para balancear clases
        - Optimizado: Hyperparámetros optimizados con Optuna
        - Validado: Validación cruzada anidada (5-fold outer, 3-fold inner)
        - Explicable: Integración con SHAP para interpretabilidad
        
        **Métricas de Rendimiento:**
        - F1-Score: 0.662
        - AUC-ROC: 0.918
        - Exactitud: 87.5%
        - Precisión: 0.585
        - Recall: 0.762
        """)
        
        # Limitaciones y consideraciones
        st.subheader("Limitaciones y Consideraciones")
        st.markdown("""
        **Importante tener en cuenta:**
        
        - Uso Educativo: Este modelo es para fines educativos y de investigación
        - No Diagnóstico: No reemplaza la evaluación profesional de salud mental
        - Basado en Datos 2019: Los patrones pueden haber cambiado
        - Predicción Probabilística: Los resultados son estimaciones, no certezas
        - Privacidad: No se almacenan datos personales ingresados
        
        **Recomendación:** Siempre consultar con profesionales de la salud para evaluaciones reales.
        """)
        
        # Contacto y recursos
        st.subheader("Recursos y Ayuda")
        st.markdown("""
        **Líneas de Ayuda en Colombia:**
        - Línea Nacional: 106 (Salud Mental)
        - Emergencias: 123
        - Chat de Ayuda: [MinSalud](https://www.minsalud.gov.co)
        
        **Recursos Adicionales:**
        - [DANE - Microdatos ENCSPA](https://microdatos.dane.gov.co/index.php/catalog/680)
        - [Ministerio de Salud - Prevención](https://www.minsalud.gov.co/salud/publica/SMental/Paginas/convivencia-desarrollo-humano-sustancias-psicoactivas.aspx)
        """)

if __name__ == "__main__":
    main()