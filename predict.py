"""
PREDICTION FUNCTIONS
═════════════════════
Loads saved models and provides clean prediction functions.
No training code — production ready.
"""

import pickle
import numpy as np
import pandas as pd

# ── Feature definitions ────────────────────────────────────────
CROP_FEATURES  = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
PRICE_FEATURES = ["crop_encoded", "season_encoded", "area_scaled"]
FERT_FEATURES  = ["crop_type_encoded", "soil_type_encoded", "deficiency_encoded"]
PEST_FEATURES  = ["crop_encoded", "pest_type_encoded", "intensity_encoded"]

# ── Load all models and encoders ───────────────────────────────
def load_artifact(path):
    """Load a pickle file from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)

# Models
crop_model  = load_artifact("models/crop_best.pkl")
price_model = load_artifact("models/price_best.pkl")
fert_model  = load_artifact("models/fert_best.pkl")
pest_model  = load_artifact("models/pest_best.pkl")

# Scalers
crop_scaler  = load_artifact("encoders/crop_scaler.pkl")
price_scaler = load_artifact("encoders/price_scaler.pkl")

# Encoders
le_crop      = load_artifact("encoders/crop_label_encoder.pkl")       # crop names
le_fert      = load_artifact("encoders/fert_fertilizer_encoder.pkl")  # fertilizer names
le_fert_crop = load_artifact("encoders/fert_crop_encoder.pkl")        # crop → int (fertilizer model)
le_soil      = load_artifact("encoders/fert_soil_encoder.pkl")        # soil type → int
le_defic     = load_artifact("encoders/fert_deficiency_encoder.pkl")  # deficiency → int
le_pest      = load_artifact("encoders/pest_pesticide_encoder.pkl")   # pesticide names
le_pest_crop = load_artifact("encoders/pest_crop_encoder.pkl")        # crop → int (pest model)
le_pest_type = load_artifact("encoders/pest_type_encoder.pkl")        # pest type → int
oe_intensity = load_artifact("encoders/pest_intensity_encoder.pkl")   # intensity → int
le_price_crop= load_artifact("encoders/price_crop_encoder.pkl")       # crop → int (price model)
oe_season    = load_artifact("encoders/price_season_encoder.pkl")     # season → int

print("✅ All models and encoders loaded.")


# ══════════════════════════════════════════════════════════════
# 1. CROP PREDICTION — Top 5 crops with confidence
# ══════════════════════════════════════════════════════════════
def predict_top5_crops(N, P, K, temperature, humidity, ph, rainfall):
    """
    Predict top 5 suitable crops with confidence scores.

    Returns:
        list of dicts: [{"crop": str, "confidence": float}, ...]
    """
    # Build scaled input DataFrame
    raw = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                       columns=CROP_FEATURES)
    scaled = pd.DataFrame(crop_scaler.transform(raw), columns=CROP_FEATURES)

    # Get probability for each class
    proba  = crop_model.predict_proba(scaled)[0]
    top5_idx = np.argsort(proba)[::-1][:5]

    results = []
    for idx in top5_idx:
        results.append({
            "crop":       le_crop.classes_[idx],
            "confidence": round(float(proba[idx]) * 100, 2),
        })
    return results


# ══════════════════════════════════════════════════════════════
# 2. PRICE PREDICTION
# ══════════════════════════════════════════════════════════════
def predict_price(crop_name, season, area):
    """
    Predict crop price (returns original scale value).

    Args:
        crop_name : str  e.g. "rice"
        season    : str  e.g. "kharif"
        area      : int  farm area (1–10)

    Returns:
        float: predicted price
    """
    # Encode inputs
    crop_enc    = le_price_crop.transform([crop_name.lower()])[0]
    season_enc  = int(oe_season.transform([[season.lower()]])[0][0])

    # Scale area (price_scaler was fit on [price, area])
    # We need area_scaled only — use a dummy price=0
    dummy       = pd.DataFrame([[0, area]], columns=["price", "area"])
    scaled_vals = price_scaler.transform(dummy)
    area_scaled = scaled_vals[0][1]

    # Predict (output is price_scaled)
    X = pd.DataFrame([[crop_enc, season_enc, area_scaled]],
                     columns=PRICE_FEATURES)
    price_scaled = price_model.predict(X)[0]

    # Inverse-transform to get actual price
    dummy_inv   = pd.DataFrame([[price_scaled, area_scaled]], columns=["price", "area"])
    actual_price = price_scaler.inverse_transform(dummy_inv)[0][0]

    return round(float(actual_price), 2)


# ══════════════════════════════════════════════════════════════
# 3. FERTILIZER RECOMMENDATION
# ══════════════════════════════════════════════════════════════
def recommend_fertilizer(crop_name, soil_type, deficiency_level):
    """
    Recommend fertilizer based on crop, soil type, and deficiency.

    Args:
        crop_name       : str  e.g. "rice"
        soil_type       : str  "clay" | "loamy" | "sandy"
        deficiency_level: str  "nitrogen" | "phosphorus" | "potassium"

    Returns:
        str: fertilizer name
    """
    crop_enc  = le_fert_crop.transform([crop_name.lower()])[0]
    soil_enc  = le_soil.transform([soil_type.lower()])[0]
    defic_enc = le_defic.transform([deficiency_level.lower()])[0]

    X = pd.DataFrame([[crop_enc, soil_enc, defic_enc]],
                     columns=FERT_FEATURES)
    pred = fert_model.predict(X)[0]
    return le_fert.inverse_transform([pred])[0]


# ══════════════════════════════════════════════════════════════
# 4. PESTICIDE RECOMMENDATION
# ══════════════════════════════════════════════════════════════
def recommend_pesticide(crop_name, pest_type, intensity):
    """
    Recommend pesticide based on crop, pest type, and intensity.

    Args:
        crop_name : str  e.g. "rice"
        pest_type : str  "aphid" | "worm" | "bug" | "caterpillar" | "whitefly"
        intensity : str  "low" | "medium" | "high"

    Returns:
        str: pesticide name
    """
    crop_enc  = le_pest_crop.transform([crop_name.lower()])[0]
    pest_enc  = le_pest_type.transform([pest_type.lower()])[0]
    intens_enc= int(oe_intensity.transform([[intensity.lower()]])[0][0])

    X = pd.DataFrame([[crop_enc, pest_enc, intens_enc]],
                     columns=PEST_FEATURES)
    pred = pest_model.predict(X)[0]
    return le_pest.inverse_transform([pred])[0]