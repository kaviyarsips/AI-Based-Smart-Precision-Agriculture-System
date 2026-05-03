"""
SMART FARMER DECISION PIPELINE
════════════════════════════════
Integrates all 4 models into one structured output.
Input  : soil (N, P, K, pH) + weather (temp, humidity, rainfall)
Output : structured dictionary with all recommendations
"""

from predict import (
    predict_top5_crops,
    predict_price,
    recommend_fertilizer,
    recommend_pesticide,
)

# Default values for pipeline extras
DEFAULT_SEASON = "kharif"
DEFAULT_AREA   = 5

# Rule-based deficiency detector from N, P, K values
def detect_deficiency(N, P, K):
    """Detect the most likely soil deficiency from N, P, K levels."""
    levels = {"nitrogen": N, "phosphorus": P, "potassium": K}
    return min(levels, key=levels.get)  # lowest value = most deficient

# Default soil type (can be extended to accept as input)
DEFAULT_SOIL = "loamy"


def run_pipeline(N, P, K, temperature, humidity, ph, rainfall,
                 season=DEFAULT_SEASON, area=DEFAULT_AREA,
                 soil_type=DEFAULT_SOIL, pest_type="worm", intensity="medium"):
    """
    Full Smart Farmer Decision Pipeline.

    Args:
        N, P, K       : float — soil nutrient ratios
        temperature   : float — °C
        humidity      : float — %
        ph            : float — soil pH
        rainfall      : float — mm/year
        season        : str   — kharif | rabi | zaid
        area          : int   — farm area in acres
        soil_type     : str   — clay | loamy | sandy
        pest_type     : str   — aphid | worm | bug | caterpillar | whitefly
        intensity     : str   — low | medium | high

    Returns:
        dict with all recommendations
    """

    result = {}

    # ── Step 1: Top 5 crop predictions ────────────────────────
    top5 = predict_top5_crops(N, P, K, temperature, humidity, ph, rainfall)
    result["top5_crops"] = top5

    # ── Step 2: Best crop (highest confidence) ─────────────────
    best_crop = top5[0]["crop"]
    result["best_crop"]       = best_crop
    result["best_confidence"] = top5[0]["confidence"]

    # ── Step 3: Price prediction for best crop ─────────────────
    try:
        price = predict_price(best_crop, season, area)
    except Exception:
        price = None  # crop may not be in price encoder
    result["predicted_price"] = price
    result["season"]          = season
    result["area_acres"]      = area

    # ── Step 4: Fertilizer recommendation ──────────────────────
    deficiency = detect_deficiency(N, P, K)
    try:
        fertilizer = recommend_fertilizer(best_crop, soil_type, deficiency)
    except Exception:
        fertilizer = "N/A"
    result["detected_deficiency"]  = deficiency
    result["soil_type"]            = soil_type
    result["fertilizer"]           = fertilizer

    # ── Step 5: Pesticide recommendation ───────────────────────
    try:
        pesticide = recommend_pesticide(best_crop, pest_type, intensity)
    except Exception:
        pesticide = "N/A"
    result["pest_type"]  = pest_type
    result["intensity"]  = intensity
    result["pesticide"]  = pesticide

    return result


# ── Quick test ─────────────────────────────────────────────────
if __name__ == "__main__":
    output = run_pipeline(
        N=90, P=42, K=43,
        temperature=20.8, humidity=82.0,
        ph=6.5, rainfall=202.9,
        season="kharif", area=5,
        soil_type="loamy",
        pest_type="worm", intensity="medium"
    )

    print("\n" + "="*55)
    print("  SMART FARMER DECISION OUTPUT")
    print("="*55)
    print(f"\n  Top 5 Crops:")
    for i, c in enumerate(output["top5_crops"], 1):
        print(f"    {i}. {c['crop']:<15} {c['confidence']}%")

    print(f"\n  Best Crop        : {output['best_crop']}  ({output['best_confidence']}%)")
    print(f"  Predicted Price  : ₹{output['predicted_price']}")
    print(f"  Deficiency       : {output['detected_deficiency']}")
    print(f"  Fertilizer       : {output['fertilizer']}")
    print(f"  Pesticide        : {output['pesticide']}")
    print("="*55)