"""
MODEL TRAINING — pesticide_clean.csv
Target: predict pesticide_encoded (classification)
"""
import pickle, time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

FEATURES = ["crop_encoded", "pest_type_encoded", "intensity_encoded"]
TARGET   = "pesticide_encoded"

df      = pd.read_csv("cleaned/pesticide_clean.csv")
X       = df[FEATURES]
y       = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

models = {
    "LogisticRegression" : LogisticRegression(max_iter=1000, random_state=42),
    "DecisionTree"       : DecisionTreeClassifier(max_depth=10, random_state=42),
    "RandomForest"       : RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "GradientBoosting"   : GradientBoostingClassifier(n_estimators=150, random_state=42),
    "KNN"                : KNeighborsClassifier(n_neighbors=5),
}

print(f"\n{'Model':<22} {'Time(s)':>8}")
print("-" * 32)
for name, model in models.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"{name:<22} {round(time.time()-t0,2):>8}")
    pickle.dump(model, open(f"models/pest_{name}.pkl", "wb"))

X_train_df = X_train.copy(); X_train_df[TARGET] = y_train.values
X_test_df  = X_test.copy();  X_test_df[TARGET]  = y_test.values
X_train_df.to_csv("cleaned/pesticide_train.csv", index=False)
X_test_df.to_csv("cleaned/pesticide_test.csv",   index=False)

print("\n✅ pesticide models saved → models/pest_*.pkl")