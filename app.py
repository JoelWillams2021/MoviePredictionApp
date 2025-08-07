# app.py

import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ─── Helpers ───────────────────────────────────────────────────────────────────

def parse_money(s: pd.Series) -> pd.Series:
    """Convert strings like '$7,000,000', '690K', '1.2M', '2B' into floats."""
    s = s.astype(str)
    s = s.str.replace(r'\(.*\)', '', regex=True)
    s = s.str.replace(r'[\$,]', '', regex=True)
    s = s.str.replace(r'^([0-9\.]+)K$', lambda m: str(float(m.group(1)) * 1e3), regex=True)
    s = s.str.replace(r'^([0-9\.]+)M$', lambda m: str(float(m.group(1)) * 1e6), regex=True)
    s = s.str.replace(r'^([0-9\.]+)B$', lambda m: str(float(m.group(1)) * 1e9), regex=True)
    return pd.to_numeric(s, errors='coerce')

# ─── App & Model Loading ───────────────────────────────────────────────────────

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Load the trained pipelines
box_model    = joblib.load('data/box_office_model.pkl')
critic_model = joblib.load('data/critic_score_model.pkl')
oscar_model  = joblib.load('data/oscar_wins_model_balanced.pkl')

# Load full dataset for recommendations
df = pd.read_csv('data/final_dataset.csv')

df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_year'] = df['release_date'].dt.year

# Preprocess numeric columns for ROI
df['budget']                = parse_money(df['budget'])
df['opening_weekend_gross'] = parse_money(df['opening_weekend_gross'])
df['gross_worldwide']       = parse_money(df['gross_worldwide'])
df = df[df['budget'] > 0].copy()
df['ROI']                   = df['gross_worldwide'] / df['budget']

# Extract some convenience fields
df['primary_genre'] = df['genres'].fillna('').str.split(',', expand=True)[0].str.strip()
df['director']      = df['directors'].fillna('').apply(lambda x: x.split(',')[0].strip())
df['cast_list']     = df['stars'].fillna('').apply(
    lambda x: [c.strip() for c in x.strip("[]").replace("'", "").split(',')][:3]
)

# ─── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # derive release_month & primary_genre if needed
    if 'release_date' in data and 'release_month' not in data:
        data['release_month'] = pd.to_datetime(data['release_date']).month_name()
    if 'genre' in data and 'primary_genre' not in data:
        data['primary_genre'] = data['genre'].split(',')[0].strip()

    df_in = pd.DataFrame([data])

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
        if feat not in df_in.columns:
            return jsonify({'error': f'Missing feature: {feat}'}), 400

    # run predictions
    box_pred    = float(box_model.predict(df_in[box_features])[0])
    critic_pred = float(critic_model.predict(df_in[critic_features])[0])
    oscar_pred  = int(oscar_model.predict(df_in[oscar_features])[0])

    # verdict logic
    budget = float(data.get('budget', 0))
    if box_pred >= 3 * budget and critic_pred >= 70:
        verdict = "👍 Strong buy – this looks like a safe investment."
    else:
        verdict = "👎 Hold off – it likely won’t meet your 3× & 70-score threshold."

    return jsonify({
        'predicted_box_office':    round(box_pred,  2),
        'predicted_critic_score':  round(critic_pred, 2),
        'predicted_oscar_wins':    oscar_pred,
        'verdict': verdict
    })


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    genre = (data.get('genre') or data.get('primary_genre') or '').strip()
    if not genre:
        return jsonify({'error': 'Missing genre'}), 400

    current_year = pd.Timestamp.now().year
    min_year = current_year - 15

    # substring match anywhere in the original comma-list
    mask = df['genres'].str.contains(genre, case=False, na=False)
    mask &= df['release_year'] >= min_year  # filter last 15 years

    candidates = df[mask]
    if candidates.empty:
        return jsonify([])

    top3 = candidates.sort_values('ROI', ascending=False).head(3)
    results = []
    for _, r in top3.iterrows():
        results.append({
            'title':    r['title'],
            'director': r['director'],
            'cast':     r['cast_list'],
            'roi':      round(r['ROI'], 2)
        })
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)