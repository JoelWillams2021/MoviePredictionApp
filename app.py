# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# 1) LOAD YOUR MODELS ONCE AT STARTUP
box_model    = joblib.load('data/box_office_model.pkl')
critic_model = joblib.load('data/critic_score_model.pkl')
oscar_model  = joblib.load('data/oscar_wins_model.pkl')

# 2) DEFINE /predict ENDPOINT
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # 2a) Derive release_month if frontend sends 'release_date'
    if 'release_date' in data and 'release_month' not in data:
        data['release_month'] = pd.to_datetime(data['release_date']).month_name()

    # 2b) Wrap into a DataFrame
    df = pd.DataFrame([data])

    # 2c) Feature lists must match what you used in training
    box_features = [
        'budget',
        'duration',
        'rating',
        'votes',
        'meta_score',
        'opening_weekend_gross',
        'MPA',
        'primary_genre',
        'release_month'
    ]
    critic_features = [
        'rating',
        'votes',
        'budget',
        'duration',
        'MPA',
        'primary_genre',
        'release_month'
    ]
    oscar_features = [
        'nominations',
        'rating',
        'votes',
        'meta_score',
        'primary_genre',
        'release_month',
        'MPA'
    ]

    # 2d) Quick sanity check
    for feat in set(box_features + critic_features + oscar_features):
        if feat not in df.columns:
            return jsonify({'error': f'Missing feature: {feat}'}), 400

    # 3) RUN PREDICTIONS
    box_pred    = box_model.predict(df[box_features])[0]
    critic_pred = critic_model.predict(df[critic_features])[0]
    oscar_pred  = oscar_model.predict(df[oscar_features])[0]

    # 4) RETURN JSON
    return jsonify({
        'predicted_box_office':    round(float(box_pred), 2),
        'predicted_critic_score':  round(float(critic_pred), 2),
        'predicted_oscar_wins':    int(oscar_pred)
    })

# 5) RUN THE APP
if __name__ == '__main__':
    # By default, runs on http://127.0.0.1:5000
    app.run(debug=True)
