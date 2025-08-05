# app.py
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# Serve everything in the project root as static (index.html, script/app.js, etc.)
app = Flask(
    __name__,
    static_folder='.',      # your project root
    static_url_path=''      # at the web root
)
CORS(app)

# 1) Load your models
box_model    = joblib.load('data/box_office_model.pkl')
critic_model = joblib.load('data/critic_score_model.pkl')
oscar_model  = joblib.load('data/oscar_wins_model_balanced.pkl')

# 2) Serve index.html at “/”
@app.route('/')
def index():
    return app.send_static_file('index.html')

# 3) Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # derive release_month if only release_date was sent
    if 'release_date' in data and 'release_month' not in data:
        data['release_month'] = pd.to_datetime(
            data['release_date']
        ).month_name()

    # derive primary_genre from the single-genre input
    if 'genre' in data and 'primary_genre' not in data:
        data['primary_genre'] = data['genre'].split(',')[0].strip()

    # wrap into a DataFrame
    df = pd.DataFrame([data])

    # feature lists (must exactly match what you used in training)
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
        'budget',
        'opening_weekend_gross',
        'duration',
        'rating',
        'votes',
        'director_avg_meta',
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

    # sanity check
    missing = [f for f in set(box_features + critic_features + oscar_features) if f not in df.columns]
    if missing:
        return jsonify({'error': f'Missing feature(s): {", ".join(missing)}'}), 400

    # run predictions
    box_pred    = box_model.predict(df[box_features])[0]
    critic_pred = critic_model.predict(df[critic_features])[0]
    oscar_pred  = oscar_model.predict(df[oscar_features])[0]

    # final verdict: 3× budget AND critic ≥70
    budget = float(data.get('budget', 0))
    if box_pred >= 3 * budget and critic_pred >= 70:
        verdict = "👍 Strong buy – this looks like a safe investment."
    else:
        verdict = "👎 Hold off – it likely won’t meet your 3× & 70-score threshold."

    return jsonify({
        'predicted_box_office':   round(float(box_pred), 2),
        'predicted_critic_score': round(float(critic_pred), 2),
        'predicted_oscar_wins':   int(oscar_pred),
        'verdict': verdict
    })

# 4) Run on host=0.0.0.0 and the PORT that Railway (or your env) specifies
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)