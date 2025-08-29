import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import json

# Load and preprocess data
with open('data/training_data.json', 'r') as f:
    data = json.load(f)
df = pd.DataFrame(data)

# Feature engineering
df['season'] = df['season'].map({'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4})  # Encode seasons
X = df[['ad_spend', 'prev_sales', 'season']]  # Features
y = df['target_sales']  # Target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
predictions = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Save model and scaler
joblib.dump(model, 'sales_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Save feature importance
feature_importance = dict(zip(X.columns, model.feature_importances_))
with open('feature_importance.json', 'w') as f:
    json.dump(feature_importance, f)

print("Model and scaler saved. Feature importance exported.")