"""
MODEL EVALUATION — pesticide models
"""
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score

FEATURES    = ["crop_encoded", "pest_type_encoded", "intensity_encoded"]
TARGET      = "pesticide_encoded"
MODEL_NAMES = ["LogisticRegression", "DecisionTree", "RandomForest", "GradientBoosting", "KNN"]

df_test  = pd.read_csv("cleaned/pesticide_test.csv")
X_test   = df_test[FEATURES]
y_test   = df_test[TARGET]
le       = pickle.load(open("encoders/pest_pesticide_encoder.pkl", "rb"))

print(f"{'Model':<22} {'Accuracy':>9} {'F1':>8}")
print("-" * 42)

results = {}
for name in MODEL_NAMES:
    model  = pickle.load(open(f"models/pest_{name}.pkl", "rb"))
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="weighted")
    results[name] = (acc, f1, model)
    print(f"{name:<22} {acc:>9.4f} {f1:>8.4f}")

best = max(results, key=lambda k: results[k][1])
print(f"\n🏆 Best: {best}  (F1={results[best][1]:.4f})")
print("\nClassification Report:\n")
best_model = results[best][2]
y_pred_best = best_model.predict(X_test)
print(classification_report(y_test, y_pred_best, target_names=le.classes_))

df_full = pd.read_csv("cleaned/pesticide_clean.csv")
cv = cross_val_score(best_model, df_full[FEATURES], df_full[TARGET], cv=5)
print(f"5-Fold CV: {cv.round(4)}  Mean: {cv.mean():.4f}")

pickle.dump(best_model, open("models/pest_best.pkl", "wb"))
print(f"\n✅ Best pesticide model saved → models/pest_best.pkl")