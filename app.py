from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(
    __name__,
    static_folder='static',    # ← this is the folder we’ll serve files from
    static_url_path=''         # ← so "/" maps to static/index.html
)

# 1) load your models once:
box_model    = joblib.load('data/box_office_model.pkl')
critic_model = joblib.load('data/critic_score_model.pkl')
oscar_model  = joblib.load('data/oscar_wins_model_balanced.pkl')

# 2) serve index.html on "/"
@app.route('/')
def home():
    return app.send_static_file('index.html')

# 3) your existing /predict endpoint…
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # derive release_month
    if 'release_date' in data and 'release_month' not in data:
        data['release_month'] = pd.to_datetime(data['release_date']).month_name()

    df = pd.DataFrame([data])

    box_features = [
        'budget','duration','rating','votes',
        'meta_score','opening_weekend_gross',
        'MPA','primary_genre','release_month'
    ]
    critic_features = [
        'budget','opening_weekend_gross','duration',
        'rating','votes','director_avg_meta',
        'MPA','primary_genre','release_month'
    ]
    oscar_features = [
        'nominations','rating','votes','meta_score',
        'primary_genre','release_month','MPA'
    ]

    # sanity check
    for feat in set(box_features + critic_features + oscar_features):
        if feat not in df.columns:
            return jsonify({'error': f'Missing feature: {feat}'}), 400

    box_pred    = box_model.predict(df[box_features])[0]
    critic_pred = critic_model.predict(df[critic_features])[0]
    oscar_pred  = oscar_model.predict(df[oscar_features])[0]

    budget = float(data.get('budget', 0))
    if box_pred >= 3 * budget and critic_pred >= 70:
        verdict = "👍 Strong buy – this looks like a safe investment."
    else:
        verdict = "👎 Hold off – it likely won’t meet your 3× & 70-score threshold."

    return jsonify({
        'predicted_box_office':    round(float(box_pred), 2),
        'predicted_critic_score':  round(float(critic_pred), 2),
        'predicted_oscar_wins':    int(oscar_pred),
        'verdict': verdict
    })

if __name__ == '__main__':
    app.run(debug=True)