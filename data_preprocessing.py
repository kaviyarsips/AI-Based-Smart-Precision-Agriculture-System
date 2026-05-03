"""
PREPROCESSING — 4 Separate Datasets
══════════════════════════════════════
crop.csv | crop_price.csv | fertilizer.csv | pesticide.csv
Each is cleaned and saved independently. No merging.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder

os.makedirs("cleaned", exist_ok=True)
os.makedirs("encoders", exist_ok=True)

# ════════════════════════════════════════════════
# 1. CROP.CSV
# ════════════════════════════════════════════════
print("\n" + "="*55)
print("  [1] PREPROCESSING: crop.csv")
print("="*55)

df = pd.read_csv("dataset/crop.csv")
print(f"  Raw shape        : {df.shape}")
print(f"  Missing values   : {df.isnull().sum().sum()}")
print(f"  Duplicates       : {df.duplicated().sum()}")

# No missing, no duplicates — standardise numerics
CROP_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

scaler_crop = StandardScaler()
df[CROP_FEATURES] = scaler_crop.fit_transform(df[CROP_FEATURES])

# Encode label
le_crop = LabelEncoder()
df["label_encoded"] = le_crop.fit_transform(df["label"])

print(f"  Crops found      : {sorted(df['label'].unique())}")
print(f"  Label mapping    : { dict(zip(le_crop.classes_, le_crop.transform(le_crop.classes_))) }")
print(f"  Final shape      : {df.shape}")

df.to_csv("cleaned/crop_clean.csv", index=False)
pickle.dump(scaler_crop, open("encoders/crop_scaler.pkl", "wb"))
pickle.dump(le_crop,     open("encoders/crop_label_encoder.pkl", "wb"))
print("  ✅ Saved → cleaned/crop_clean.csv")


# ════════════════════════════════════════════════
# 2. CROP_PRICE.CSV
# ════════════════════════════════════════════════
print("\n" + "="*55)
print("  [2] PREPROCESSING: crop_price.csv")
print("="*55)

df = pd.read_csv("dataset/crop_price.csv")
print(f"  Raw shape        : {df.shape}")
print(f"  Missing values   : {df.isnull().sum().sum()}")
print(f"  Duplicates       : {df.duplicated().sum()}")

# Remove duplicates
before = len(df)
df = df.drop_duplicates()
print(f"  After dedup      : {len(df)} rows (removed {before - len(df)})")

# Standardise text
df["crop"]   = df["crop"].str.strip().str.lower()
df["season"] = df["season"].str.strip().str.lower()

# Encode crop
le_price_crop = LabelEncoder()
df["crop_encoded"] = le_price_crop.fit_transform(df["crop"])

# Encode season with ordinal (natural order)
season_order = [["zaid", "kharif", "rabi"]]
oe_season = OrdinalEncoder(categories=season_order)
df["season_encoded"] = oe_season.fit_transform(df[["season"]]).astype(int)

# Scale price and area
scaler_price = StandardScaler()
df[["price_scaled", "area_scaled"]] = scaler_price.fit_transform(df[["price", "area"]])

# Outlier check on price (IQR)
Q1, Q3 = df["price"].quantile(0.25), df["price"].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df["price"] < Q1 - 1.5*IQR) | (df["price"] > Q3 + 1.5*IQR)]
print(f"  Price outliers   : {len(outliers)} rows (kept — valid market variation)")
print(f"  Price range      : {df['price'].min()} – {df['price'].max()}")
print(f"  Seasons          : {sorted(df['season'].unique())}")
print(f"  Final shape      : {df.shape}")

df.to_csv("cleaned/crop_price_clean.csv", index=False)
pickle.dump(le_price_crop, open("encoders/price_crop_encoder.pkl", "wb"))
pickle.dump(oe_season,     open("encoders/price_season_encoder.pkl", "wb"))
pickle.dump(scaler_price,  open("encoders/price_scaler.pkl", "wb"))
print("  ✅ Saved → cleaned/crop_price_clean.csv")


# ════════════════════════════════════════════════
# 3. FERTILIZER.CSV
# ════════════════════════════════════════════════
print("\n" + "="*55)
print("  [3] PREPROCESSING: fertilizer.csv")
print("="*55)

df = pd.read_csv("dataset/fertilizer.csv")
print(f"  Raw shape        : {df.shape}")
print(f"  Missing values   : {df.isnull().sum().sum()}")
print(f"  Duplicates       : {df.duplicated().sum()}")

# Duplicates are valid here — same crop/soil/deficiency can repeat
# Drop full exact duplicates only
before = len(df)
df = df.drop_duplicates()
print(f"  After dedup      : {len(df)} rows (removed {before - len(df)})")

# Standardise text
for col in ["crop_type", "soil_type", "deficiency_level", "fertilizer"]:
    df[col] = df[col].str.strip().str.lower()

# Encode each categorical column
le_fert_crop  = LabelEncoder()
le_soil       = LabelEncoder()
le_deficiency = LabelEncoder()
le_fertilizer = LabelEncoder()

df["crop_type_encoded"]      = le_fert_crop.fit_transform(df["crop_type"])
df["soil_type_encoded"]      = le_soil.fit_transform(df["soil_type"])
df["deficiency_encoded"]     = le_deficiency.fit_transform(df["deficiency_level"])
df["fertilizer_encoded"]     = le_fertilizer.fit_transform(df["fertilizer"])

print(f"  Crop types       : {sorted(df['crop_type'].unique())}")
print(f"  Soil types       : {sorted(df['soil_type'].unique())}")
print(f"  Deficiency types : {sorted(df['deficiency_level'].unique())}")
print(f"  Fertilizers      : {sorted(df['fertilizer'].unique())}")
print(f"  Final shape      : {df.shape}")

df.to_csv("cleaned/fertilizer_clean.csv", index=False)
pickle.dump(le_fert_crop,  open("encoders/fert_crop_encoder.pkl",  "wb"))
pickle.dump(le_soil,       open("encoders/fert_soil_encoder.pkl",  "wb"))
pickle.dump(le_deficiency, open("encoders/fert_deficiency_encoder.pkl", "wb"))
pickle.dump(le_fertilizer, open("encoders/fert_fertilizer_encoder.pkl", "wb"))
print("  ✅ Saved → cleaned/fertilizer_clean.csv")


# ════════════════════════════════════════════════
# 4. PESTICIDE.CSV
# ════════════════════════════════════════════════
print("\n" + "="*55)
print("  [4] PREPROCESSING: pesticide.csv")
print("="*55)

df = pd.read_csv("dataset/pesticide.csv")
print(f"  Raw shape        : {df.shape}")
print(f"  Missing values   : {df.isnull().sum().sum()}")
print(f"  Duplicates       : {df.duplicated().sum()}")

before = len(df)
df = df.drop_duplicates()
print(f"  After dedup      : {len(df)} rows (removed {before - len(df)})")

# Standardise text
for col in ["crop", "pest_type", "intensity", "pesticide"]:
    df[col] = df[col].str.strip().str.lower()

# Encode intensity with natural order (low < medium < high)
intensity_order = [["low", "medium", "high"]]
oe_intensity = OrdinalEncoder(categories=intensity_order)
df["intensity_encoded"] = oe_intensity.fit_transform(df[["intensity"]]).astype(int)

# Encode other categoricals
le_pest_crop  = LabelEncoder()
le_pest_type  = LabelEncoder()
le_pesticide  = LabelEncoder()

df["crop_encoded"]      = le_pest_crop.fit_transform(df["crop"])
df["pest_type_encoded"] = le_pest_type.fit_transform(df["pest_type"])
df["pesticide_encoded"] = le_pesticide.fit_transform(df["pesticide"])

print(f"  Crops            : {sorted(df['crop'].unique())}")
print(f"  Pest types       : {sorted(df['pest_type'].unique())}")
print(f"  Intensity levels : {sorted(df['intensity'].unique())}")
print(f"  Pesticides       : {sorted(df['pesticide'].unique())}")
print(f"  Final shape      : {df.shape}")

df.to_csv("cleaned/pesticide_clean.csv", index=False)
pickle.dump(le_pest_crop,  open("encoders/pest_crop_encoder.pkl",     "wb"))
pickle.dump(le_pest_type,  open("encoders/pest_type_encoder.pkl",     "wb"))
pickle.dump(oe_intensity,  open("encoders/pest_intensity_encoder.pkl","wb"))
pickle.dump(le_pesticide,  open("encoders/pest_pesticide_encoder.pkl","wb"))
print("  ✅ Saved → cleaned/pesticide_clean.csv")


print("\n" + "="*55)
print("  ✅ ALL 4 DATASETS PREPROCESSED SUCCESSFULLY")
print("  Cleaned files → cleaned/")
print("  Encoders      → encoders/")
print("="*55)