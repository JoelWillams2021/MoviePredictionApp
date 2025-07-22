# train_box_office_model.py

import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

def parse_money(s: pd.Series) -> pd.Series:
    # Remove parenthetical notes, $ and commas
    s = s.str.replace(r'\(.*\)', '', regex=True)
    s = s.str.replace(r'[\$,]', '', regex=True)
    # Convert "7M" / "7.5M" / "7B" suffixes
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
    # Extract hours and minutes and compute total minutes
    hours = s.str.extract(r'(\d+)\s*h', expand=False).fillna(0).astype(int)
    mins  = s.str.extract(r'(\d+)\s*m', expand=False).fillna(0).astype(int)
    return hours * 60 + mins

# 1. LOAD DATA
df = pd.read_csv('data/final_dataset.csv')
print(f"Initial shape: {df.shape}")

# 2. FEATURE ENGINEER
df['release_date']   = pd.to_datetime(df['release_date'], errors='coerce')
df['release_month']  = df['release_date'].dt.month_name()
df['primary_genre']  = df['genres'].fillna('').str.split(',', expand=True)[0].str.strip()

# 3. PARSE / CLEAN SPECIFIC COLUMNS
df['budget']                 = parse_money(df['budget'].astype(str))
df['opening_weekend_gross']  = parse_money(df['opening_weekend_gross'].astype(str))
df['gross_worldwide']        = parse_money(df['gross_worldwide'].astype(str))
df['duration']               = parse_duration(df['duration'].astype(str))
df['rating']                 = pd.to_numeric(df['rating'], errors='coerce')
df['votes']                  = parse_money(df['votes'].astype(str))  # "690K" → 690000
df['meta_score']             = pd.to_numeric(df['meta_score'], errors='coerce')

# 4. DEFINE FEATURES & TARGET
numeric_features = [
    'budget',
    'duration',
    'rating',
    'votes',
    'meta_score',
    'opening_weekend_gross'
]
categorical_features = [
    'MPA',
    'primary_genre',
    'release_month'
]
TARGET = 'gross_worldwide'

# 5. HANDLE MISSING DATA
print("\nNull counts before cleaning:")
print(df[numeric_features + [TARGET]].isnull().sum())

# Drop rows missing target
before = len(df)
df = df.dropna(subset=[TARGET])
after = len(df)
print(f"Dropped {before - after} rows missing '{TARGET}'; {after} remain.")

# Impute numeric with median
df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())

# Fill categoricals
df['MPA']           = df['MPA'].fillna('NR')
df['primary_genre'] = df['primary_genre'].fillna('Unknown')
df['release_month'] = df['release_month'].fillna('Unknown')

print("\nNull counts after imputation:")
feat_cols = numeric_features + categorical_features
print(df[feat_cols].isnull().sum())

if df.shape[0] == 0:
    raise RuntimeError("No data left after cleaning! Check your parsing logic.")

# 6. PREPARE X & y
X = df[feat_cols]
y = df[TARGET]

# 7. BUILD PIPELINE
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 8. SPLIT & TRAIN
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

model.fit(X_train, y_train)

# 9. SAVE MODEL
joblib.dump(model, 'Data/box_office_model.pkl')
print("✅ box_office_model.pkl saved in data/")
