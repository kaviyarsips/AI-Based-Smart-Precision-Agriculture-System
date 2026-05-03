# ══════════════════════════════════════════════════════════════
# REAL PIPELINE — connected to your trained models
# ═══════════
import streamlit as st
import time
# ═══════════════════════════════════════════════════
import pickle
import numpy as np
import pandas as pd

# Load once at startup
crop_model   = pickle.load(open("models/crop_best.pkl",  "rb"))
crop_scaler  = pickle.load(open("encoders/crop_scaler.pkl", "rb"))
le_crop      = pickle.load(open("encoders/crop_label_encoder.pkl", "rb"))
price_model  = pickle.load(open("models/price_best.pkl", "rb"))
price_scaler = pickle.load(open("encoders/price_scaler.pkl", "rb"))
le_price_crop= pickle.load(open("encoders/price_crop_encoder.pkl", "rb"))
oe_season    = pickle.load(open("encoders/price_season_encoder.pkl", "rb"))
fert_model   = pickle.load(open("models/fert_best.pkl",  "rb"))
le_fert      = pickle.load(open("encoders/fert_fertilizer_encoder.pkl", "rb"))
le_fert_crop = pickle.load(open("encoders/fert_crop_encoder.pkl", "rb"))
le_soil      = pickle.load(open("encoders/fert_soil_encoder.pkl", "rb"))
le_defic     = pickle.load(open("encoders/fert_deficiency_encoder.pkl", "rb"))
pest_model   = pickle.load(open("models/pest_best.pkl",  "rb"))
le_pest      = pickle.load(open("encoders/pest_pesticide_encoder.pkl", "rb"))
le_pest_crop = pickle.load(open("encoders/pest_crop_encoder.pkl", "rb"))
le_pest_type = pickle.load(open("encoders/pest_type_encoder.pkl", "rb"))
oe_intensity = pickle.load(open("encoders/pest_intensity_encoder.pkl", "rb"))

CROP_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

CROP_EMOJIS = {
    "rice":"🌾","wheat":"🌿","maize":"🌽","cotton":"🌸","millet":"🌱",
    "chickpea":"🫘","banana":"🍌","mango":"🥭","coffee":"☕","coconut":"🥥",
    "orange":"🍊","apple":"🍎","grapes":"🍇","watermelon":"🍉","papaya":"🍈",
    "pomegranate":"🍎","jute":"🌿","kidneybeans":"🫘","lentil":"🫘",
    "blackgram":"🫘","mungbean":"🌱","mothbeans":"🌱","pigeonpeas":"🌿",
}

def detect_deficiency(N, P, K):
    return min({"nitrogen": N, "phosphorus": P, "potassium": K}, key=lambda k: {"nitrogen":N,"phosphorus":P,"potassium":K}[k])

def smart_recommendation(inp):
    # ── Step 1: Top 5 crops ──────────────────────────────────
    raw    = pd.DataFrame([[inp["N"], inp["P"], inp["K"],
                            inp["temperature"], inp["humidity"],
                            inp["ph"], inp["rainfall"]]],
                          columns=CROP_FEATURES)
    scaled = pd.DataFrame(crop_scaler.transform(raw), columns=CROP_FEATURES)

    proba  = crop_model.predict_proba(scaled)[0]
    top5   = np.argsort(proba)[::-1][:5]

    top5_crops = []
    for idx in top5:
        name = le_crop.classes_[idx]
        top5_crops.append({
            "crop":       name,
            "confidence": round(float(proba[idx]) * 100, 2),
            "emoji":      CROP_EMOJIS.get(name, "🌱"),
        })

    best_crop = top5_crops[0]["crop"]

    # ── Step 2: Price ────────────────────────────────────────
    try:
        crop_enc   = le_price_crop.transform([best_crop])[0]
        season_enc = int(oe_season.transform([[inp["season"]]])[0][0])
        dummy      = pd.DataFrame([[0, inp["area"]]], columns=["price","area"])
        area_scaled= price_scaler.transform(dummy)[0][1]
        X_price    = pd.DataFrame([[crop_enc, season_enc, area_scaled]],
                                  columns=["crop_encoded","season_encoded","area_scaled"])
        price_scaled = price_model.predict(X_price)[0]
        dummy_inv    = pd.DataFrame([[price_scaled, area_scaled]], columns=["price","area"])
        price = round(float(price_scaler.inverse_transform(dummy_inv)[0][0]), 2)
    except Exception:
        price = None

    # ── Step 3: Fertilizer ───────────────────────────────────
    deficiency = detect_deficiency(inp["N"], inp["P"], inp["K"])
    try:
        X_fert = pd.DataFrame([[
            le_fert_crop.transform([best_crop])[0],
            le_soil.transform([inp["soil_type"]])[0],
            le_defic.transform([deficiency])[0],
        ]], columns=["crop_type_encoded","soil_type_encoded","deficiency_encoded"])
        fertilizer = le_fert.inverse_transform(fert_model.predict(X_fert))[0]
    except Exception:
        fertilizer = "N/A"

    # ── Step 4: Pesticide ────────────────────────────────────
    try:
        X_pest = pd.DataFrame([[
            le_pest_crop.transform([best_crop])[0],
            le_pest_type.transform([inp["pest_type"]])[0],
            int(oe_intensity.transform([[inp["intensity"]]])[0][0]),
        ]], columns=["crop_encoded","pest_type_encoded","intensity_encoded"])
        pesticide = le_pest.inverse_transform(pest_model.predict(X_pest))[0]
    except Exception:
        pesticide = "N/A"

    return {
        "top5_crops":          top5_crops,
        "best_crop":           best_crop,
        "best_confidence":     top5_crops[0]["confidence"],
        "predicted_price":     price,
        "season":              inp["season"],
        "area_acres":          inp["area"],
        "detected_deficiency": deficiency,
        "soil_type":           inp["soil_type"],
        "fertilizer":          fertilizer,
        "pest_type":           inp["pest_type"],
        "intensity":           inp["intensity"],
        "pesticide":           pesticide,
    }
# ══════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=DM+Sans:wght@300;400;500&display=swap');

    /* ── Reset & Base ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif !important;
    }

    /* ── Animated gradient background ── */
    .stApp {
        background: linear-gradient(-45deg, #d8f3dc, #f0faf2, #b7e4c7, #e8f8ed, #95d5b2, #f5fcf7);
        background-size: 400% 400%;
        animation: gradientShift 14s ease infinite;
        min-height: 100vh;
    }

    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding: 2rem 3rem !important;
        max-width: 1100px !important;
    }

    /* ── Leaf pattern overlay (SVG via CSS) ── */
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        background-image:
            radial-gradient(circle at 15% 25%, rgba(52,168,83,0.06) 0%, transparent 45%),
            radial-gradient(circle at 85% 75%, rgba(52,168,83,0.05) 0%, transparent 45%),
            radial-gradient(circle at 50% 50%, rgba(255,255,255,0.3) 0%, transparent 70%);
        pointer-events: none;
        z-index: 0;
    }

    /* ── Glass card base ── */
    .glass-card {
        background: rgba(255, 255, 255, 0.55);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid rgba(255,255,255,0.75);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(52,168,83,0.1), 0 2px 8px rgba(0,0,0,0.04);
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
    }

    /* ── Hero title ── */
    .hero-title {
        font-family: 'Playfair Display', serif !important;
        font-size: 3.2rem;
        font-weight: 700;
        color: #1a3a0a;
        text-align: center;
        line-height: 1.15;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 12px rgba(52,168,83,0.12);
    }

    .hero-sub {
        font-size: 1.1rem;
        color: #3a6b28;
        text-align: center;
        font-weight: 300;
        margin-bottom: 2.5rem;
        letter-spacing: 0.01em;
    }

    /* ── Section headings ── */
    .section-heading {
        font-family: 'Playfair Display', serif !important;
        font-size: 1.4rem;
        font-weight: 700;
        color: #1a3a0a;
        margin-bottom: 1.2rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(149,213,178,0.5);
    }

    /* ── Crop cards ── */
    .crop-card {
        background: rgba(255,255,255,0.6);
        backdrop-filter: blur(14px);
        border: 1.5px solid rgba(149,213,178,0.5);
        border-radius: 16px;
        padding: 1.25rem 1rem;
        text-align: center;
        transition: transform 0.25s, box-shadow 0.25s, border-color 0.25s;
        box-shadow: 0 4px 16px rgba(52,168,83,0.08);
        height: 100%;
    }

    .crop-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 32px rgba(52,168,83,0.18);
        border-color: rgba(52,168,83,0.5);
    }

    .crop-card.best {
        background: rgba(216,243,220,0.75);
        border: 2.5px solid #52b788;
        box-shadow: 0 8px 28px rgba(52,168,83,0.2);
    }

    .crop-emoji   { font-size: 2.8rem; line-height: 1; margin-bottom: 8px; }
    .crop-name    { font-family: 'Playfair Display', serif; font-size: 1.2rem; font-weight: 700; color: #1a3a0a; margin-bottom: 4px; }
    .crop-conf    { font-size: 1.5rem; font-weight: 700; color: #2d6a4f; margin-bottom: 4px; }
    .crop-label   { font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; color: #74a98a; }
    .best-badge   {
        display: inline-block;
        background: #52b788; color: white;
        font-size: 10px; font-weight: 700;
        letter-spacing: 0.1em; text-transform: uppercase;
        padding: 3px 10px; border-radius: 100px;
        margin-bottom: 8px;
    }

    /* ── Confidence bar ── */
    .conf-bar-bg {
        height: 5px; border-radius: 3px;
        background: rgba(149,213,178,0.3);
        margin-top: 8px; overflow: hidden;
    }
    .conf-bar-fill {
        height: 100%; border-radius: 3px;
        background: linear-gradient(90deg, #74c69d, #52b788);
        transition: width 1s ease;
    }

    /* ── Result metric cards ── */
    .metric-card {
        background: rgba(255,255,255,0.6);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(149,213,178,0.5);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 4px 16px rgba(52,168,83,0.08);
        transition: transform 0.2s, box-shadow 0.2s;
        text-align: center;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(52,168,83,0.15);
    }
    .metric-icon  { font-size: 2rem; margin-bottom: 6px; }
    .metric-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; color: #74a98a; margin-bottom: 4px; }
    .metric-value { font-family: 'Playfair Display', serif; font-size: 1.6rem; font-weight: 700; color: #1a3a0a; }
    .metric-sub   { font-size: 12px; color: #5a8a6a; margin-top: 2px; }

    /* ── Input styling ── */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div,
    .stSlider { border-radius: 10px !important; }

    div[data-testid="stNumberInput"] input {
        background: rgba(255,255,255,0.7) !important;
        border: 1.5px solid rgba(149,213,178,0.6) !important;
        border-radius: 10px !important;
        color: #1a3a0a !important;
        font-size: 15px !important;
    }

    div[data-testid="stNumberInput"] input:focus {
        border-color: #52b788 !important;
        box-shadow: 0 0 0 3px rgba(82,183,136,0.15) !important;
    }

    .stSelectbox > div > div {
        background: rgba(255,255,255,0.7) !important;
        border: 1.5px solid rgba(149,213,178,0.6) !important;
        border-radius: 10px !important;
        color: #1a3a0a !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #52b788, #2d6a4f) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        height: 52px !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 4px 18px rgba(52,183,136,0.3) !important;
        transition: all 0.2s !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #74c69d, #40916c) !important;
        box-shadow: 0 6px 24px rgba(52,183,136,0.45) !important;
        transform: translateY(-2px) !important;
    }

    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* Back button variant */
    .back-btn .stButton > button {
        background: rgba(255,255,255,0.6) !important;
        color: #2d6a4f !important;
        border: 1.5px solid rgba(82,183,136,0.4) !important;
        box-shadow: none !important;
        width: auto !important;
    }

    /* ── Input labels ── */
    .stNumberInput label, .stSelectbox label, .stSlider label {
        color: #2d6a4f !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
    }

    /* ── Slider ── */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #74c69d, #52b788) !important;
    }

    /* ── Divider ── */
    hr { border-color: rgba(149,213,178,0.3) !important; }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: #52b788 !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #f0faf2; }
    ::-webkit-scrollbar-thumb { background: #95d5b2; border-radius: 3px; }

    /* ── Fade-in animation ── */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .fade-in { animation: fadeUp 0.5s ease both; }
    .fade-in-1 { animation: fadeUp 0.5s ease 0.1s both; }
    .fade-in-2 { animation: fadeUp 0.5s ease 0.2s both; }
    .fade-in-3 { animation: fadeUp 0.5s ease 0.3s both; }
    .fade-in-4 { animation: fadeUp 0.5s ease 0.4s both; }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "page":    "home",
        "results": None,
        "inputs":  {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ══════════════════════════════════════════════════════════════
# PAGE 1 — HOME / INPUT
# ══════════════════════════════════════════════════════════════
def page_home():
    # ── Hero ──────────────────────────────────────────────────
    st.markdown("""
    <div class="fade-in" style="padding-top:1rem">
      <div class="hero-title">🌾 Smart Farmer Assistant</div>
      <p class="hero-sub">
        AI-powered precision agriculture — know your best crop, price & care plan before you sow
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Input card ────────────────────────────────────────────
    st.markdown('<div class="glass-card fade-in-1">', unsafe_allow_html=True)

    # Soil nutrients
    st.markdown('<div class="section-heading">Soil Nutrients</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    N  = c1.number_input(" Nitrogen (N) — mg/kg",   min_value=0,   max_value=200, value=90,  step=1)
    P  = c2.number_input(" Phosphorus (P) — mg/kg", min_value=0,   max_value=200, value=42,  step=1)
    K  = c3.number_input(" Potassium (K) — mg/kg",  min_value=0,   max_value=200, value=43,  step=1)

    st.markdown("<br>", unsafe_allow_html=True)

    # Soil & climate
    st.markdown('<div class="section-heading"> Soil & Climate</div>', unsafe_allow_html=True)
    c4, c5 = st.columns(2)
    ph          = c4.number_input("Soil pH",           min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    temperature = c5.number_input("Temperature (°C)",  min_value=-5.0, max_value=55.0, value=20.8, step=0.5)

    c6, c7 = st.columns(2)
    humidity = c6.slider(" Humidity (%)",      min_value=10,   max_value=100, value=82)
    rainfall = c7.slider(" Rainfall (mm/yr)",  min_value=0,    max_value=3000, value=202)

    st.markdown("<br>", unsafe_allow_html=True)

    # Farm details
    st.markdown('<div class="section-heading">🌍 Farm Details</div>', unsafe_allow_html=True)
    c8, c9, c10, c11 = st.columns(4)
    season    = c8.selectbox(" Season",       ["kharif", "rabi", "zaid"])
    area      = c9.number_input(" Area (acres)", min_value=1, max_value=100, value=5)
    soil_type = c10.selectbox(" Soil Type",   ["loamy", "clay", "sandy"])
    pest_type = c11.selectbox(" Pest Type",   ["worm", "aphid", "bug", "caterpillar", "whitefly"])
    intensity = st.radio(" Pest Intensity", ["low", "medium", "high"], horizontal=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Submit button ─────────────────────────────────────────
    _, center, _ = st.columns([1, 2, 1])
    with center:
        clicked = st.button("Analyze 🌱", use_container_width=True)

    if clicked:
        inputs = dict(
            N=N, P=P, K=K, ph=ph, temperature=temperature,
            humidity=humidity, rainfall=rainfall,
            season=season, area=area, soil_type=soil_type,
            pest_type=pest_type, intensity=intensity,
        )
        with st.spinner("🌱 Analysing your farm data…"):
            time.sleep(1.2)  # visual delay — remove for production
            results = smart_recommendation(inputs)

        st.session_state["inputs"]  = inputs
        st.session_state["results"] = results
        st.session_state["page"]    = "results"
        st.rerun()


# ══════════════════════════════════════════════════════════════
# PAGE 2 — RESULTS
# ══════════════════════════════════════════════════════════════
CROP_EMOJIS = {
    "rice":"🌾","wheat":"🌿","maize":"🌽","cotton":"🌸","millet":"🌱",
    "barley":"🌾","chickpea":"🫘","banana":"🍌","mango":"🥭","coffee":"☕",
    "coconut":"🥥","orange":"🍊","apple":"🍎","grapes":"🍇","watermelon":"🍉",
    "papaya":"🍈","pomegranate":"🍎","jute":"🌿","kidneybeans":"🫘",
    "lentil":"🫘","blackgram":"🫘","mungbean":"🌱","mothbeans":"🌱","pigeonpeas":"🌿",
}

def get_emoji(crop):
    return CROP_EMOJIS.get(crop.lower(), "🌱")

def page_results():
    out    = st.session_state["results"]
    inputs = st.session_state["inputs"]

    if not out:
        st.session_state["page"] = "home"
        st.rerun()

    # ── Back button ───────────────────────────────────────────
    st.markdown('<div class="back-btn">', unsafe_allow_html=True)
    if st.button("⬅ Back to Input"):
        st.session_state["page"] = "home"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Page title ────────────────────────────────────────────
    st.markdown("""
    <div class="fade-in" style="text-align:center;padding:1rem 0 0.5rem">
      <div class="hero-title" style="font-size:2.4rem">🌱 Your Farm Recommendations</div>
      <p class="hero-sub">AI analysis complete — here's your personalised farming plan</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Best crop hero ────────────────────────────────────────
    best  = out["best_crop"]
    conf  = out["best_confidence"]
    emoji = get_emoji(best)

    st.markdown(f"""
    <div class="glass-card fade-in-1" style="text-align:center;padding:2rem">
      <div style="font-size:4rem;margin-bottom:10px">{emoji}</div>
      <div class="best-badge">✨ Best Match</div>
      <div style="font-family:'Playfair Display',serif;font-size:2.2rem;font-weight:700;color:#1a3a0a;margin:6px 0">{best.title()}</div>
      <div style="font-size:1.1rem;color:#5a8a6a">Confidence Score</div>
      <div style="font-family:'Playfair Display',serif;font-size:3rem;font-weight:700;color:#2d6a4f;line-height:1.1">{conf}%</div>
      <div style="max-width:300px;margin:12px auto 0">
        <div class="conf-bar-bg">
          <div class="conf-bar-fill" style="width:{conf}%"></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Top 5 crops ───────────────────────────────────────────
    st.markdown('<div class="glass-card fade-in-2">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">🏆 Top 5 Crop Recommendations</div>', unsafe_allow_html=True)

    top5  = out["top5_crops"]
    cols  = st.columns(5)
    medals= ["1-Suggested","2-Consider","3-Better One","4️- Avoid ","5️Better Avoid"]

    for i, (col, crop) in enumerate(zip(cols, top5)):
        is_best  = (i == 0)
        em       = crop.get("emoji", get_emoji(crop["crop"]))
        card_cls = "crop-card best" if is_best else "crop-card"
        badge    = '<div class="best-badge">Best Choice</div>' if is_best else ""
        with col:
            st.markdown(f"""
            <div class="{card_cls}" style="animation:fadeUp 0.5s ease {0.05*i}s both">
              {badge}
              <div class="crop-emoji">{em}</div>
              <div class="crop-name">{crop['crop'].title()}</div>
              <div class="crop-conf">{crop['confidence']}%</div>
              <div class="crop-label">Confidence</div>
              <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{crop['confidence']}%"></div>
              </div>
              <div style="font-size:1.4rem;margin-top:8px">{medals[i]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Metric cards row ──────────────────────────────────────
    st.markdown('<div class="fade-in-3">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading"> Your Complete Farm Plan</div>', unsafe_allow_html=True)

    price = out.get("predicted_price")
    price_str = f"₹{price:,.0f}" if price else "N/A"

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-icon"></div>
          <div class="metric-label">Predicted Price</div>
          <div class="metric-value">{price_str}</div>
          <div class="metric-sub">{out.get('season','').title()} · {out.get('area_acres','')} acres</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-icon">🌿</div>
          <div class="metric-label">Fertilizer</div>
          <div class="metric-value">{out.get('fertilizer','N/A').title()}</div>
          <div class="metric-sub">Deficiency: {out.get('detected_deficiency','').title()}</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-icon"></div>
          <div class="metric-label">Pesticide</div>
          <div class="metric-value">{out.get('pesticide','N/A')}</div>
          <div class="metric-sub">{out.get('pest_type','').title()} · {out.get('intensity','').title()}</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-icon">🪨</div>
          <div class="metric-label">Soil Type</div>
          <div class="metric-value">{out.get('soil_type','N/A').title()}</div>
          <div class="metric-sub">Season: {out.get('season','').title()}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Input summary ─────────────────────────────────────────
    with st.expander(" View Your Input Parameters"):
        ic1, ic2, ic3, ic4 = st.columns(4)
        ic1.metric("Nitrogen",    f"{inputs.get('N','-')} mg/kg")
        ic2.metric("Phosphorus",  f"{inputs.get('P','-')} mg/kg")
        ic3.metric("Potassium",   f"{inputs.get('K','-')} mg/kg")
        ic4.metric("pH",          f"{inputs.get('ph','-')}")
        ic5, ic6, ic7, _ = st.columns(4)
        ic5.metric("Temperature", f"{inputs.get('temperature','-')} °C")
        ic6.metric("Humidity",    f"{inputs.get('humidity','-')} %")
        ic7.metric("Rainfall",    f"{inputs.get('rainfall','-')} mm")

    # ── Footer ────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:2rem 0 1rem;color:#74a98a;font-size:12px">
      Smart Farmer Assistant · AI-Powered Precision Agriculture
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════
def main():
    init_state()
    inject_css()

    if st.session_state["page"] == "home":
        page_home()
    elif st.session_state["page"] == "results":
        page_results()

if __name__ == "__main__":
    main()