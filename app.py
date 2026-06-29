import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="SAT-ML SCM", layout="wide", page_icon="🚚")

# --- SISTEMA DE LOGIN ---
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        st.markdown("<h1 style='text-align: center;'>🔒 Acceso SAT-ML SCM</h1>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            usuario = st.text_input("Usuario")
            clave = st.text_input("Contraseña", type="password")
            if st.button("Iniciar Sesión"):
                if usuario == "admin" and clave == "admin123":
                    st.session_state["password_correct"] = True
                    st.rerun()
                else:
                    st.error("Credenciales incorrectas.")
        return False
    return True

if check_password():
    # --- CARGA DE MODELOS ---
    @st.cache_resource
    def load_artifacts():
        try:
            model = joblib.load('models/rf_model.pkl')
            scaler = joblib.load('models/scaler.pkl')
            columnas = joblib.load('models/columnas.pkl')
            return model, scaler, columnas
        except:
            return None, None, None

    model, scaler, columnas_entrenamiento = load_artifacts()

    # --- MENÚ LATERAL ---
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2760/2760205.png", width=100)
    st.sidebar.title("Menú SAT-ML")
    menu = st.sidebar.radio("Navegación:", ["📊 Dashboard General", "🚨 Predicción de Retrasos", "📈 Pronóstico de Demanda"])

    # --- PESTAÑA 1: DASHBOARD ---
    if menu == "📊 Dashboard General":
        st.title("📊 Visión Global de la Operación Logística")
        try:
            df = pd.read_csv('data/historico_cedis.csv')
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Despachos", f"{len(df):,}")
            retrasos = df['Retraso'].sum()
            c2.metric("Tasa de Retraso", f"{(retrasos/len(df))*100:.1f}%")
            c3.metric("Nivel de Servicio (OTIF)", f"{100 - (retrasos/len(df))*100:.1f}%")

            st.markdown("### Análisis de Cuellos de Botella (Capacidad vs Tiempo)")
            # Gráfico de dispersión optimizado
            fig = px.scatter(df, x="Capacidad_Muelle", y="Tiempo_Cargue", color="Retraso", 
                             opacity=0.6, facet_col="CEDI", hover_data=["Volumen_Carga"],
                             color_continuous_scale="RdYlGn_r")
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.warning("No se encontró 'data/historico_cedis.csv'.")

    # --- PESTAÑA 2: PREDICCIÓN ---
    elif menu == "🚨 Predicción de Retrasos":
        st.title("🚨 Simulador de Riesgo de Retraso")
        
        if model:
            with st.form("pred_form"):
                col1, col2 = st.columns(2)
                with col1:
                    cedi = st.selectbox("Seleccione CEDI", ["Tenjo", "Soacha"])
                    volumen = st.number_input("Volumen de Carga (kg)", value=5000)
                    capacidad = st.slider("Saturación Capacidad Muelle (%)", 10, 100, 85)
                    tiempo = st.number_input("Tiempo Estimado de Cargue (min)", value=60)
                with col2:
                    clima = st.selectbox("Condición Climática", ["Despejado", "Lluvia", "Tormenta"])
                    trafico = st.selectbox("Tráfico en Ruta", ["Bajo", "Moderado", "Alto"])
                    distancia = st.number_input("Distancia Ruta (km)", value=50)
                    ventana = st.selectbox("Ventana de Entrega", ["Mañana", "Tarde", "Noche"])
                
                submit = st.form_submit_button("Analizar Riesgo")

            if submit:
                # Crear diccionario con los inputs
                input_dict = {
                    'Volumen_Carga': volumen,
                    'Capacidad_Muelle': capacidad,
                    'Tiempo_Cargue': tiempo,
                    'Distancia_Ruta': distancia,
                    f'CEDI_{cedi}': 1,
                    f'Clima_{clima}': 1,
                    f'Trafico_{trafico}': 1,
                    f'Ventana_Entrega_{ventana}': 1
                }
                
                # Alinear con las columnas de entrenamiento
                input_df = pd.DataFrame(columns=columnas_entrenamiento)
                input_df.loc[0] = 0 # Llenar con ceros por defecto
                
                for key, val in input_dict.items():
                    if key in input_df.columns:
                        input_df.at[0, key] = val

                # Escalar y predecir
                input_scaled = scaler.transform(input_df)
                prob = model.predict_proba(input_scaled)[0][1]

                st.markdown("---")
                prob_pct = prob * 100
                if prob > 0.7:
                    st.error(f"🔴 ALERTA ROJA: Riesgo Crítico de Retraso ({prob_pct:.1f}%).")
                elif prob > 0.4:
                    st.warning(f"🟡 ALERTA AMARILLA: Riesgo Moderado ({prob_pct:.1f}%).")
                else:
                    st.success(f"🟢 ALERTA VERDE: Despacho Seguro ({prob_pct:.1f}%).")
        else:
            st.error("Modelos no encontrados. Ejecuta 'train.py' primero.")

    # --- PESTAÑA 3: PRONÓSTICO (Con Botón de Descarga) ---
    elif menu == "📈 Pronóstico de Demanda":
        st.title("📈 Pronóstico Inteligente de Demanda (Prophet)")
        
        forecast_path = 'data/demanda_forecast.csv'
        if os.path.exists(forecast_path):
            df_forecast = pd.read_csv(forecast_path)
            
            fig = px.line(df_forecast, x='ds', y='yhat', title='Proyección de Volumen a 30 Días')
            fig.add_scatter(x=df_forecast['ds'], y=df_forecast['yhat_upper'], mode='lines', line=dict(dash='dot'), name='Límite Superior')
            fig.add_scatter(x=df_forecast['ds'], y=df_forecast['yhat_lower'], mode='lines', line=dict(dash='dot'), name='Límite Inferior')
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 📥 Descargar Datos de Pronóstico")
            csv = df_forecast.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Descargar CSV para integración ERP",
                data=csv,
                file_name='SATML_Pronostico.csv',
                mime='text/csv'
            )
        else:
            st.info("No se encontró el pronóstico. Ejecuta 'forecasting.py' primero.")