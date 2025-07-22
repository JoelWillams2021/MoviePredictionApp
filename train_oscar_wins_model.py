
#!/usr/bin/env python3
# train_oscar_wins_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import re

def parse_money(s: pd.Series) -> pd.Series:
    """Convert strings like '690K', '1.1M', '2B', '$1,200,000' into floats."""
    s = s.astype(str)
    s = s.str.replace(r'\(.*\)', '', regex=True)
    s = s.str.replace(r'[\$,]', '', regex=True)
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

# 1. LOAD DATA
df_movies = pd.read_csv('data/final_dataset.csv')
df_oscars = pd.read_csv('data/the_oscar_award.csv')

# 2. AGGREGATE NOMINATIONS & WINS
df_oscars['nominations'] = 1
df_oscars['wins'] = df_oscars['winner'].astype(bool).astype(int)
agg = (
    df_oscars
    .groupby('film', as_index=False)
    .agg(nominations=('nominations','sum'),
         wins=('wins','sum'))
)

# 3. MERGE WITH MOVIE METADATA
df = df_movies.merge(
    agg,
    left_on='title',
    right_on='film',
    how='left'
)

# missing nominations/wins -> 0
df['nominations'] = df['nominations'].fillna(0).astype(int)
df['wins']        = df['wins'].fillna(0).astype(int)

# 4. FEATURE ENGINEERING
df['release_date']  = pd.to_datetime(df['release_date'], errors='coerce')
df['release_month'] = df['release_date'].dt.month_name()
df['primary_genre'] = df['genres'].fillna('').str.split(',', expand=True)[0].str.strip()
df['MPA']           = df['MPA'].fillna('NR')

# 5. CLEAN & PARSE NUMERIC COLUMNS
df['rating']     = pd.to_numeric(df['rating'], errors='coerce')
df['votes']      = parse_money(df['votes'])
df['meta_score'] = pd.to_numeric(df['meta_score'], errors='coerce')

# 6. DROP ROWS MISSING KEY FEATURES
needed = ['nominations','rating','votes','meta_score','primary_genre','release_month','MPA']
df = df.dropna(subset=['nominations','rating','votes','meta_score'])

# 7. IMPUTE NUMERICS & FILL CATEGORICALS
numeric_feats = ['nominations','rating','votes','meta_score']
df[numeric_feats] = df[numeric_feats].fillna(df[numeric_feats].median())
df['primary_genre'] = df['primary_genre'].fillna('Unknown')
df['release_month'] = df['release_month'].fillna('Unknown')
df['MPA']           = df['MPA'].fillna('NR')

# 8. PREP X & y
feature_cols = numeric_feats + ['primary_genre','release_month','MPA']
X = df[feature_cols]
y = df['wins'].astype(int)

# 9. BUILD PIPELINE
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_feats),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['primary_genre','release_month','MPA'])
])
oscar_model = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 10. SPLIT & TRAIN
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
oscar_model.fit(X_train, y_train)

# 11. SAVE MODEL (compressed)
joblib.dump(oscar_model, 'data/oscar_wins_model.pkl', compress=('gzip', 3))
print("✅ oscar_wins_model.pkl saved in data/ (gzip‑3 compressed)")
