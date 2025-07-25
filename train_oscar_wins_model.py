# train_oscar_wins_model_improved.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import re

# Helper to parse numeric strings into floats
def parse_money(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    # remove parenthetical notes
    s = s.str.replace(r"\(.*\)", "", regex=True)
    # strip $ and commas
    s = s.str.replace(r"[\$,]", "", regex=True)
    # Handle K, M, B suffixes
    s = s.str.replace(r"^([0-9\.]+)K$", lambda m: str(float(m.group(1)) * 1e3), regex=True)
    s = s.str.replace(r"^([0-9\.]+)M$", lambda m: str(float(m.group(1)) * 1e6), regex=True)
    s = s.str.replace(r"^([0-9\.]+)B$", lambda m: str(float(m.group(1)) * 1e9), regex=True)
    return pd.to_numeric(s, errors='coerce')

# 1. LOAD DATA
df_movies = pd.read_csv('data/final_dataset.csv')
df_oscars = pd.read_csv('data/the_oscar_award.csv')

# 2. AGGREGATE
agg = (
    df_oscars
    .assign(nominations=1, wins=df_oscars['winner'].astype(int))
    .groupby('film', as_index=False)
    .agg(nominations=('nominations','sum'), wins=('wins','sum'))
)

# 3. MERGE
df = df_movies.merge(
    agg,
    left_on='title',
    right_on='film',
    how='left'
)
df['nominations'] = df['nominations'].fillna(0).astype(int)
df['wins']        = df['wins'].fillna(0).astype(int)

# 4. FEATURE ENGINEERING
df['release_date']  = pd.to_datetime(df['release_date'], errors='coerce')
df['release_month'] = df['release_date'].dt.month_name()
df['primary_genre'] = df['genres'].fillna('').str.split(',', expand=True)[0].str.strip()
df['MPA']           = df['MPA'].fillna('NR')

# 5. CLEAN NUMERICS
df['rating']     = pd.to_numeric(df['rating'], errors='coerce')
df['votes']      = parse_money(df['votes'])
df['meta_score'] = pd.to_numeric(df['meta_score'], errors='coerce')

# 6. DROP MISSING
df = df.dropna(subset=['nominations','rating','votes','meta_score'])

# 7. IMPUTE & FILL
numeric_feats = ['nominations','rating','votes','meta_score']
df[numeric_feats] = df[numeric_feats].fillna(df[numeric_feats].median())
df['primary_genre'] = df['primary_genre'].fillna('Unknown')
df['release_month'] = df['release_month'].fillna('Unknown')
df['MPA']           = df['MPA'].fillna('NR')

# 8. PREP X & y
categorical_feats = ['primary_genre','release_month','MPA']
X = df[numeric_feats + categorical_feats]
y = df['wins']
print(f"Classes distribution:\n{y.value_counts(normalize=True).sort_index()}\n")

# 9. PREPROCESSOR
num_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])
cat_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer([
    ('num', num_pipe, numeric_feats),
    ('cat', cat_pipe, categorical_feats)
])

# 10. BUILD IMB PIPELINE WITH RANDOMOVERSAMPLER + CLASSIFIER
pipeline = ImbPipeline([
    ('pre', preprocessor),
    ('ros', RandomOverSampler(random_state=42)),
    ('clf', RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=42))
])

# 11. SPLIT & TRAIN
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train/Test split: {len(X_train)}/{len(X_test)} samples.")

pipeline.fit(X_train, y_train)

# 12. EVALUATE
y_pred = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Weighted F1: {f1_score(y_test, y_pred, average='weighted'):.3f}")
print("Classification report:")
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# 13. SAVE
joblib.dump(pipeline, 'data/oscar_wins_model_balanced.pkl', compress=('gzip',3))
print("✅ Saved balanced Oscar wins model to data/oscar_wins_model_balanced.pkl")
