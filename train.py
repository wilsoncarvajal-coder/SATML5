import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def entrenar_modelo():
    print("⚙️ Iniciando entrenamiento del modelo SAT-ML...")
    
    # 1. Cargar datos
    df = pd.read_csv('data/historico_cedis.csv')
    
    # 2. Preprocesamiento
    # Asegurarnos de que 'Retraso' es la variable objetivo
    X = df.drop(columns=['Retraso', 'ID_Despacho', 'Fecha'], errors='ignore')
    y = df['Retraso']
    
    # Convertir variables categóricas (One-Hot Encoding)
    X_encoded = pd.get_dummies(X, drop_first=True)
    columnas_entrenamiento = X_encoded.columns.tolist()
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    # Escalar numéricos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 3. Entrenar Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # 4. Guardar artefactos en la carpeta 'models/'
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf_model, 'models/rf_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(columnas_entrenamiento, 'models/columnas.pkl') # Vital para la App
    
    print("✅ Entrenamiento completado. Modelos guardados en la carpeta 'models/'.")

if __name__ == "__main__":
    entrenar_modelo()
