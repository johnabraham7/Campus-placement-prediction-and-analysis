from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

def load_model():
    with open('models/placement_model.pkl', 'rb') as model_file:
        model, scaler = pickle.load(model_file)
    return model, scaler

model, scaler = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    feature_names = ['Communication', 'Aptitude and Reasoning', 'CGPA', 'Certifications', 'Coding Skills', 'Hackathons', 'Coding Contests', 'Internships', 'Projects']
    thresholds = {
        'Communication': 7,
        'Aptitude and Reasoning': 8,
        'CGPA': 7.5,
        'Certifications': 2,
        'Coding Skills': 6,
        'Hackathons': 1,
        'Coding Contests': 1,
        'Internships': 1,
        'Projects': 1
    }

    features_array = np.array(features).reshape(1, -1)
    scaled_features = scaler.transform(features_array)
    prediction = model.predict(scaled_features)
    prediction_proba = model.predict_proba(scaled_features)[0][1]

    improvement_areas = []
    if prediction[0] == 0:
        for i, feature_name in enumerate(feature_names):
            if features[i] < thresholds[feature_name]:
                improvement_areas.append(feature_name)

    return render_template('result.html', prediction_proba=prediction_proba, improvement_areas=improvement_areas)

if __name__ == '__main__':
    app.run(debug=True)
