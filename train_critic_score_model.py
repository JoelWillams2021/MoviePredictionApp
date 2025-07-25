##!/usr/bin/env python3
# train_critic_score_model.py

import re
import pandas as pd
from textblob import TextBlob
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import joblib

# --- Helper functions ---

def parse_money(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(r"\(.*\)", "", regex=True)
    s = s.str.replace(r"[\$,]", "", regex=True)
    s = s.str.replace(r"^([0-9\.]+)K$", lambda m: str(float(m.group(1)) * 1e3), regex=True)
    s = s.str.replace(r"^([0-9\.]+)M$", lambda m: str(float(m.group(1)) * 1e6), regex=True)
    s = s.str.replace(r"^([0-9\.]+)B$", lambda m: str(float(m.group(1)) * 1e9), regex=True)
    return pd.to_numeric(s, errors='coerce')


def parse_duration(s: pd.Series) -> pd.Series:
    hrs = s.str.extract(r"(\d+)\s*h", expand=False).fillna(0).astype(int)
    mins = s.str.extract(r"(\d+)\s*m", expand=False).fillna(0).astype(int)
    return hrs * 60 + mins

# 1. Load data
print("Loading data...")
df = pd.read_csv('data/final_dataset.csv')
print(f"Initial rows: {len(df)}")
# drop missing
required = ['meta_score','rating','votes','budget','duration','genres','release_date','opening_weekend_gross']
df = df.dropna(subset=required)
print(f"After dropna: {len(df)} rows")

# 2. Feature engineering
# release and genres
df['release_date']  = pd.to_datetime(df['release_date'], errors='coerce')
df['release_month'] = df['release_date'].dt.month_name()
df['primary_genre'] = df['genres'].str.split(',', expand=True)[0].str.strip()
df['MPA']           = df['MPA'].fillna('NR')
# parse numeric
df['budget']                = parse_money(df['budget'])
df['opening_weekend_gross'] = parse_money(df['opening_weekend_gross'])
df['votes']                 = parse_money(df['votes'])
df['rating']                = pd.to_numeric(df['rating'], errors='coerce')
df['duration']              = parse_duration(df['duration'].astype(str))
df['meta_score']            = pd.to_numeric(df['meta_score'], errors='coerce')

# NLP on description


# Director average Metascore
df['primary_director'] = df['directors'].fillna('').str.split(',', expand=True)[0].str.strip("[] '\"")
past_meta = df.groupby('primary_director')['meta_score'].transform('mean')
df['director_avg_meta'] = past_meta.fillna(df['meta_score'].median())

# 3. Prepare features and target
numeric_features = [
    'budget', 'opening_weekend_gross', 'duration', 'rating', 'votes', 'director_avg_meta'
]
categorical_features = ['MPA', 'primary_genre', 'release_month']
target = 'meta_score'

X = df[numeric_features + categorical_features]
y = df[target]
print(f"Features shape: {X.shape}")

# 4. Preprocessing pipeline
num_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale',  StandardScaler())
])
cat_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preproc = ColumnTransformer([
    ('num', num_pipe, numeric_features),
    ('cat', cat_pipe, categorical_features)
])

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train/Test split: {len(X_train)}/{len(X_test)}")

# 6. Model & tuning
model = Pipeline([
    ('pre', preproc),
    ('reg', HistGradientBoostingRegressor(random_state=42))
])
param_dist = {
    'reg__max_iter': [100,200,500],
    'reg__learning_rate': [0.01,0.05,0.1],
    'reg__max_leaf_nodes': [20,30,40],
    'reg__l2_regularization': [0.0,0.1,1.0]
}
search = RandomizedSearchCV(
    model, param_dist, n_iter=20, cv=5,
    scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42
)
print("Running hyperparameter search...")
search.fit(X_train, y_train)
best = search.best_estimator_
print(f"Best CV MAE: {-search.best_score_:.2f}")
print(f"Best params: {search.best_params_}")

# 7. Final evaluation
y_pred = best.predict(X_test)
print(f"Final MAE:  {mean_absolute_error(y_test, y_pred):.2f}")
print(f"Final RMSE: {root_mean_squared_error(y_test, y_pred):.2f}")
print(f"Final R^2:  {r2_score(y_test, y_pred):.3f}")

# 8. Save model
dest = 'data/critic_score_model.py'
joblib.dump(best, dest.replace('.py','.pkl'), compress=('gzip',3))
print(f"✅ Saved model to {dest.replace('.py','.pkl')}")
