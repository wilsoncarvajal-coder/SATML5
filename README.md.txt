# 🚚 SAT-ML SCM: Sistema de Alerta Temprana Logística

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest%20%7C%20Prophet-brightgreen)

## 📖 Descripción del Proyecto
**SAT-ML SCM** (Sistema de Alerta Temprana con Machine Learning para Supply Chain Management) es una solución analítica integral desarrollada para predecir y mitigar riesgos operativos, retrasos y cuellos de botella en Centros de Distribución (CEDIs), específicamente enfocada en las operaciones de **Tenjo y Soacha**.

El sistema utiliza algoritmos de ensamble (Random Forest) para clasificar el riesgo de retraso de un despacho en tiempo real, y modelos de series temporales (Prophet) para pronosticar la demanda logística, permitiendo a la gerencia pasar de un modelo de gestión reactivo a uno **predictivo y proactivo**.

## ✨ Características Principales
* **🔐 Autenticación Segura:** Sistema de login para proteger la información operativa.
* **📊 Dashboard Interactivo:** Visualización de KPIs (OTIF, Tasa de Retraso) y gráficos dinámicos de cuellos de botella para evitar el *overplotting*.
* **🚨 Simulador de Riesgos:** Motor de inferencia que evalúa variables como volumen de carga, capacidad del muelle, clima y tráfico para generar alertas tempranas (Verde, Amarilla, Roja).
* **📈 Pronóstico de Demanda:** Proyección a 30 días del volumen de operación utilizando Meta Prophet, con capacidad de **exportación a CSV** para integración con sistemas WMS/ERP.

## 📂 Arquitectura del Repositorio
El proyecto sigue una arquitectura unificada "All-in-One" para facilitar el despliegue continuo (CI/CD):

```text
/
├── app.py                 # Frontend web interactivo (Streamlit)
├── train.py               # Backend: Entrenamiento y guardado del modelo Random Forest
├── forecasting.py         # Backend: Generación del pronóstico con Prophet
├── requirements.txt       # Dependencias del proyecto
├── data/                  # Almacenamiento de datasets (historico_cedis.csv, demanda_forecast.csv)
└── models/                # Artefactos exportados (.pkl) para consumo de la App