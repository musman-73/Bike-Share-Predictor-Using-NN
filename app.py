from flask import Flask, render_template, request, jsonify
import joblib
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the model and encoder
model = tf.keras.models.load_model("model.h5")
encoder = joblib.load('encoder.pkl')

def predict(data):
    encoded_data = encoder.transform(np.array(data['categorical_data']).reshape(1, -1))
    input_data = np.concatenate([np.array(data['data']).reshape(1, -1), encoded_data], axis=1)
    predictions = np.square(model.predict(input_data))
    return predictions

# Define the Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()
    predictions = predict(data)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)
