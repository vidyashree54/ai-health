# train_fitness_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import os

# ========== 1️⃣ LOAD DATA ==========
df = pd.read_csv("FitBit data.csv")

# Keep only the relevant numeric columns
cols = [
    'TotalSteps', 'TotalDistance', 'VeryActiveMinutes',
    'FairlyActiveMinutes', 'LightlyActiveMinutes',
    'SedentaryMinutes', 'Calories'
]
df = df[cols].dropna()

# Split into X, y for calorie prediction
X = df[['TotalSteps', 'TotalDistance', 'VeryActiveMinutes',
        'FairlyActiveMinutes', 'LightlyActiveMinutes', 'SedentaryMinutes']]
y = df['Calories']

# ========== 2️⃣ PREPROCESSING ==========
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# ========== 3️⃣ MODEL 1: CALORIE PREDICTION ==========
model_calorie = Sequential([
    Input(shape=(6,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model_calorie.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

es = EarlyStopping(patience=10, restore_best_weights=True)

print("\n🏋️ Training Calorie Prediction Model...")
model_calorie.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=100, batch_size=16, callbacks=[es], verbose=1)

# Evaluate
loss, mae = model_calorie.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Calorie Prediction Model trained | Test MAE: {mae:.4f}")

# ========== 4️⃣ MODEL 2: ACTIVITY RECOMMENDATION ==========
# Here we reverse the relationship
X2 = df[['Calories']]
y2 = df[['TotalSteps', 'TotalDistance', 'VeryActiveMinutes',
         'FairlyActiveMinutes', 'LightlyActiveMinutes']]

scaler_X2 = StandardScaler()
scaler_y2 = StandardScaler()

X2_scaled = scaler_X2.fit_transform(X2)
y2_scaled = scaler_y2.fit_transform(y2)

X2_train, X2_test, y2_train, y2_test = train_test_split(X2_scaled, y2_scaled, test_size=0.2, random_state=42)

model_activity = Sequential([
    Input(shape=(1,)),
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    Dense(5, activation='relu')
])

model_activity.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

print("\n💧 Training Activity Recommendation Model...")
model_activity.fit(X2_train, y2_train, validation_data=(X2_test, y2_test),
                   epochs=100, batch_size=16, callbacks=[es], verbose=1)

loss2, mae2 = model_activity.evaluate(X2_test, y2_test, verbose=0)
print(f"\n✅ Activity Recommendation Model trained | Test MAE: {mae2:.4f}")

# ========== 5️⃣ SAVE MODELS ==========
os.makedirs("models", exist_ok=True)
model_calorie.save("models/calorie_predictor.h5")
model_activity.save("models/activity_recommender.h5")

import joblib
joblib.dump(scaler_X, "models/scaler_X.pkl")
joblib.dump(scaler_y, "models/scaler_y.pkl")
joblib.dump(scaler_X2, "models/scaler_X2.pkl")
joblib.dump(scaler_y2, "models/scaler_y2.pkl")

print("\n🎯 All models and scalers saved successfully in /models folder!")

# ========== 6️⃣ EXAMPLE USAGE ==========
# Example: predict calories
sample = np.array([[8000, 5.5, 45, 20, 200, 600]]).reshape(1, -1)
sample_scaled = scaler_X.transform(sample)
pred_scaled = model_calorie.predict(sample_scaled)
pred_calories = scaler_y.inverse_transform(pred_scaled)
print(f"\n🔥 Predicted Calories for sample activity: {pred_calories[0][0]:.2f} kcal")

# Example: recommend activity for calorie goal
goal_cal = np.array([[3000]]).reshape(1, -1)
goal_scaled = scaler_X2.transform(goal_cal)
activity_scaled = model_activity.predict(goal_scaled)
activity = scaler_y2.inverse_transform(activity_scaled)
print(f"🏃 Recommended activity for 3000 kcal goal:\nSteps: {activity[0][0]:.0f}, Distance: {activity[0][1]:.2f} km, "
      f"Very Active: {activity[0][2]:.0f} min, Fairly Active: {activity[0][3]:.0f} min, Lightly Active: {activity[0][4]:.0f} min")
