# train_box_office_model.py
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import joblib

def parse_money(s: pd.Series) -> pd.Series:
    """Convert strings like '$7,000,000', '7M', '1.2B' into floats."""
    s = s.astype(str)
    s = s.str.replace(r'\(.*\)', '', regex=True)      # remove parentheticals
    s = s.str.replace(r'[\$,]', '', regex=True)        # strip $ and commas
    # Handle K, M, B suffixes
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
    """Convert '2h 4m' into total minutes."""
    hours = s.str.extract(r'(\d+)\s*h', expand=False).fillna(0).astype(int)
    mins  = s.str.extract(r'(\d+)\s*m', expand=False).fillna(0).astype(int)
    return hours * 60 + mins

# 1. LOAD DATA
print("Loading data...")
df = pd.read_csv('data/final_dataset.csv')
print(f"Initial shape: {df.shape}")

# 2. FEATURE ENGINEERING
df['release_date']  = pd.to_datetime(df['release_date'], errors='coerce')
df['release_month'] = df['release_date'].dt.month_name()
df['primary_genre'] = df['genres'].fillna('').str.split(',', expand=True)[0].str.strip()

# 3. PARSE / CLEAN SPECIFIC COLUMNS
df['budget']                = parse_money(df['budget'])
df['opening_weekend_gross'] = parse_money(df['opening_weekend_gross'])
df['gross_worldwide']       = parse_money(df['gross_worldwide'])
df['duration']              = parse_duration(df['duration'].astype(str))
df['rating']                = pd.to_numeric(df['rating'], errors='coerce')
df['votes']                 = parse_money(df['votes'].astype(str))
df['meta_score']            = pd.to_numeric(df['meta_score'], errors='coerce')

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

# 6. PIPELINE WITH IMPUTERS
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features),
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor',   RandomForestRegressor(n_estimators=100, random_state=42))
])

# 7. SPLIT & TRAIN
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X = df[numeric_features + categorical_features]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# 8. TRAIN MODEL
print("Training model...")
model.fit(X_train, y_train)

# 9. EVALUATE MODEL
print("Evaluating model performance...")
y_pred = model.predict(X_test)

# Compute metrics without deprecated arguments
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")
print(f"R^2:  {r2:.3f}")

# 10. SAVE MODEL SAVE MODEL
joblib.dump(model, 'data/box_office_model.pkl', compress=('gzip', 3))
print("✅ box_office_model.pkl saved in data/ (gzip-3 compressed)")
