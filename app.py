from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
with open('diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    age = float(request.form['Age'])

    values = {
        'Pregnancies': float(request.form['Pregnancies']),
        'Glucose': float(request.form['Glucose']),
        'BloodPressure': float(request.form['BloodPressure']),
        'SkinThickness': float(request.form['SkinThickness']),
        'Insulin': float(request.form['Insulin']),
        'BMI': float(request.form['BMI']),
        'DiabetesPedigreeFunction': float(request.form['DiabetesPedigreeFunction']),
        'Age': age
    }

    # Prepare and scale features
    features = np.array([list(values.values())])
    scaled_features = scaler.transform(features)

    probability = model.predict_proba(scaled_features)[0][1]
    result = "Diabetic" if probability >= 0.5 else "Not Diabetic"

    return render_template("report.html", name=name, values=values, result=result, probability=round(probability * 100, 2))

if __name__ == '__main__':
    app.run(debug=True)
