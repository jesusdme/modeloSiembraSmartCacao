import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Cargar el modelo desde el archivo
rf_model = joblib.load('cacao_rf_model.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Extracción de datos de la solicitud
    area_sembrada = data.get('Area_Sembrada')
    area_cosechada = data.get('Area_Cosechada')
    produccion = data.get('Produccion')
    
    # Validar datos
    if area_sembrada is None or area_cosechada is None or produccion is None:
        return jsonify({'error': 'Parámetros insuficientes'}), 400
    
    # Crear un DataFrame con los datos para realizar la predicción
    input_data = pd.DataFrame([[area_sembrada, area_cosechada, produccion]], columns=['Area_Sembrada', 'Area_Cosechada', 'Produccion'])
    # Realizar predicción
    prediction = rf_model.predict(input_data)[0]
    
    return jsonify({'Rendimiento_Predicho': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
