# train_critic_score_model.py
#!/usr/bin/env python3
# train_critic_score_model.py

import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

def parse_money(s: pd.Series) -> pd.Series:
    """Convert money strings like '$7,000,000 (estimated)', '690K', '1.2M', '2B' into floats."""
    s = s.str.replace(r'\(.*\)', '', regex=True)      # remove parenthetical notes
    s = s.str.replace(r'[\$,]', '', regex=True)        # strip $ and commas
    # Handle 'K', 'M', 'B' suffixes
    s = s.str.replace(
        r'^([0-9\.]+)K$', 
        lambda m: str(float(m.group(1)) * 1e3), 
        regex=True
    )
    s = s.str.replace(
        r'^([0-9\.]+)M$', 
        lambda m: str(float(m.group(1)) * 1e6), 
        regex=True
    )
    s = s.str.replace(
        r'^([0-9\.]+)B$', 
        lambda m: str(float(m.group(1)) * 1e9), 
        regex=True
    )
    return pd.to_numeric(s, errors='coerce')

def parse_duration(s: pd.Series) -> pd.Series:
    """Convert duration strings like '2h 4m' into total minutes."""
    hours = s.str.extract(r'(\d+)\s*h', expand=False).fillna(0).astype(int)
    mins  = s.str.extract(r'(\d+)\s*m', expand=False).fillna(0).astype(int)
    return hours * 60 + mins

# 1. LOAD YOUR DATA
df = pd.read_csv('data/final_dataset.csv')
print(f"Initial shape: {df.shape}")

# 2. DROP ROWS MISSING CRITICAL FIELDS
df = df.dropna(subset=[
    'meta_score',    # target
    'rating',
    'votes',
    'budget',
    'duration',
    'genres',
    'release_date'
])

# 3. FEATURE ENGINEERING
df['release_date']  = pd.to_datetime(df['release_date'], errors='coerce')
df['release_month'] = df['release_date'].dt.month_name()
df['primary_genre'] = (
    df['genres']
      .fillna('')
      .str.split(',', expand=True)[0]
      .str.strip()
)
df['MPA'] = df['MPA'].fillna('NR')  # Not Rated fallback

# 4. PARSE & CLEAN NUMERIC COLUMNS
df['budget']       = parse_money(df['budget'].astype(str))
df['votes']        = parse_money(df['votes'].astype(str))
df['rating']       = pd.to_numeric(df['rating'], errors='coerce')
df['meta_score']   = pd.to_numeric(df['meta_score'], errors='coerce')
df['duration']     = parse_duration(df['duration'].astype(str))

# 5. REPORT NULLS BEFORE CLEANING
numeric_features = ['budget', 'duration', 'rating', 'votes']
print("\nNull counts before cleaning:")
print(df[numeric_features + ['meta_score']].isnull().sum())

# 6. DROP ROWS MISSING TARGET
before = len(df)
df = df.dropna(subset=['meta_score'])
after = len(df)
print(f"Dropped {before - after} rows missing 'meta_score'; {after} remain.")

# 7. IMPUTE NUMERIC FEATURES
df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())

# 8. FILL CATEGORICAL FEATURES
df['MPA']           = df['MPA'].fillna('NR')
df['primary_genre'] = df['primary_genre'].fillna('Unknown')
df['release_month'] = df['release_month'].fillna('Unknown')

# 9. VERIFY NO NaNs IN FEATURES
categorical_features = ['MPA', 'primary_genre', 'release_month']
feat_cols = numeric_features + categorical_features
print("\nNull counts after imputation:")
print(df[feat_cols].isnull().sum())

if df.shape[0] == 0:
    raise RuntimeError("No data left after cleaning! Check parsing logic.")

# 10. PREPARE X & y
X = df[feat_cols]
y = df['meta_score']

# 11. BUILD PREPROCESSOR & PIPELINE
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
])
critic_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 12. SPLIT & TRAIN
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

critic_model.fit(X_train, y_train)

# 13. SAVE THE TRAINED MODEL (compressed)
joblib.dump(critic_model, 'data/critic_score_model.pkl', compress=('gzip', 3))
print("✅ critic_score_model.pkl saved in data/ (gzip-3 compressed)")           