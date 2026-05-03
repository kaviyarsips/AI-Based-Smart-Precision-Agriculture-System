"""
MODEL EVALUATION — crop_price models (regression)
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES    = ["crop_encoded", "season_encoded", "area_scaled"]
TARGET      = "price_scaled"
MODEL_NAMES = ["LinearRegression", "Ridge", "DecisionTree", "RandomForest", "GradientBoosting"]

df_test  = pd.read_csv("cleaned/crop_price_test.csv")
X_test   = df_test[FEATURES]
y_test   = df_test[TARGET]

print(f"{'Model':<22} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print("-" * 50)

results = {}
for name in MODEL_NAMES:
    model  = pickle.load(open(f"models/price_{name}.pkl", "rb"))
    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)
    results[name] = (mae, rmse, r2, model)
    print(f"{name:<22} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f}")

best = max(results, key=lambda k: results[k][2])  # best R²
print(f"\n🏆 Best: {best}  (R²={results[best][2]:.4f})")

pickle.dump(results[best][3], open("models/price_best.pkl", "wb"))
print(f"✅ Best price model saved → models/price_best.pkl")