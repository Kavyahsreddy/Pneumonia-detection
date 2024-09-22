from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Extract input values from the form                                             
        age = float(request.form['age'])
        temperature = float(request.form['temperature'])
        blood_oxygen_level = float(request.form['blood_oxygen_level'])
        respiratory_rate = float(request.form['respiratory_rate'])

        # Create feature array
        features = np.array([[age, temperature, blood_oxygen_level, respiratory_rate]])

        # Make prediction
        prediction = model.predict(features)
        result = 'Pneumonia' if prediction[0] == 1 else 'No Pneumonia'

        return render_template('result.html', result=result)
    
    return render_template('predict.html')

@app.route('/instructions')
def instructions():
    return render_template('predict.html', show_instructions=True)

@app.route('/back_to_predict')
def back_to_predict():
    return render_template('predict.html', show_instructions=False)

if __name__ == '__main__':
    app.run(debug=True)
