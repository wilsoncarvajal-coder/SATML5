import pandas as pd
import os
from prophet import Prophet

def generar_pronostico():
    print("📈 Iniciando pronóstico de demanda con Prophet...")
    
    # Cargar datos
    df = pd.read_csv('data/historico_cedis.csv')
    
    # Preparar datos para Prophet: Necesita columnas 'ds' (fecha) y 'y' (valor a predecir)
    # Agrupamos por fecha para obtener el volumen diario total
    df_diario = df.groupby('Fecha')['Volumen_Carga'].sum().reset_index()
    df_diario.columns = ['ds', 'y']
    df_diario['ds'] = pd.to_datetime(df_diario['ds'])
    
    # Entrenar modelo Prophet
    modelo_prophet = Prophet(yearly_seasonality=True, daily_seasonality=False)
    modelo_prophet.fit(df_diario)
    
    # Proyectar 30 días hacia el futuro
    futuro = modelo_prophet.make_future_dataframe(periods=30)
    forecast = modelo_prophet.predict(futuro)
    
    # Guardar pronóstico para la app
    os.makedirs('data', exist_ok=True)
    forecast.to_csv('data/demanda_forecast.csv', index=False)
    
    print("✅ Pronóstico generado y guardado en 'data/demanda_forecast.csv'.")

if __name__ == "__main__":
    generar_pronostico()