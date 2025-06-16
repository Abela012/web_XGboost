from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load your trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "ML Model is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Validate the incoming request
    features = data.get('features')
    if not features or len(features) != 24:
        return jsonify({'error': 'Expected 24 features'}), 400
    
    # Predict using the model
    prediction = model.predict([features])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
