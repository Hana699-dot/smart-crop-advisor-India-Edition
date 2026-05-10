"""
🌾 Smart Crop Advisor — India Edition (v3 Enhanced)
====================================================
Stack  : Python · Scikit-learn · XGBoost · Ensemble ML · Streamlit · ReportLab · OWM API
Model  : Voting Classifier (Random Forest + XGBoost + Decision Tree) — 62 Indian crops
Dataset: 4,960 agronomic records across all Indian states, soil types & seasons
Security: API key via st.secrets — never hardcoded
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import io, warnings, datetime
import joblib, numpy as np, pandas as pd, requests
import streamlit as st
from pathlib import Path

from reportlab.lib           import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles    import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units     import cm
from reportlab.platypus      import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "india_crop_model.pkl"

# ── Secure API Key ────────────────────────────────────────────────────────────
try:
    OWM_API_KEY = st.secrets["OWM_API_KEY"]
except (KeyError, FileNotFoundError):
    OWM_API_KEY = None
OWM_URL = "https://api.openweathermap.org/data/2.5/weather"


# ══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE — 62 Indian Crops
# Tuple: (category, emoji, season, profit, water_need, fertilizer,
#         disease_risk, sow_month, harvest_days, why_text, msp_per_quintal)
# ══════════════════════════════════════════════════════════════════════════════
CROP_DB = {
    # CEREALS & MILLETS
    "rice":        ("cereal","🌾","Kharif (Jun–Nov)","₹35,000–₹55,000/acre","High (600–1200mm)","NPK 80:40:40; Urea top-dress at tillering","Blast, BLB, Sheath rot","Jun–Jul",120,"Thrives in waterlogged fields with warm temps & heavy rain.",2183),
    "wheat":       ("cereal","🌿","Rabi (Oct–Mar)","₹28,000–₹48,000/acre","Moderate (4–6 irrigations)","NPK 120:60:40; Split urea in 3 doses","Rust, Powdery mildew","Oct–Nov",120,"Cool dry winters are ideal; alluvial plains of Punjab-UP give best yield.",2275),
    "maize":       ("cereal","🌽","Kharif / Rabi","₹25,000–₹45,000/acre","Moderate (500–700mm)","NPK 120:60:60; Zinc sulphate 25kg/ha","Downy mildew, Stalk rot","Jun–Jul / Oct",90,"Versatile crop for warm climates with well-drained soils.",1962),
    "jowar":       ("millet","🌾","Kharif / Rabi","₹18,000–₹35,000/acre","Low (400–600mm)","NPK 60:30:30","Anthracnose, Shoot fly","Jun / Oct",110,"Drought-tolerant; excellent for semi-arid black soils of Maharashtra.",3180),
    "bajra":       ("millet","🌻","Kharif (Jun–Sep)","₹15,000–₹30,000/acre","Very Low (250–400mm)","NPK 40:20:20","Downy mildew","Jun–Jul",80,"Thrives in hot dry conditions; sandy soils of Rajasthan.",2500),
    "ragi":        ("millet","🌱","Kharif (Jun–Oct)","₹18,000–₹32,000/acre","Moderate (500–800mm)","NPK 50:25:25","Blast, Brown leaf spot","Jun–Jul",130,"Highly nutritious millet; ideal for red & laterite soils of Karnataka.",3846),
    "barley":      ("cereal","🌾","Rabi (Oct–Mar)","₹20,000–₹35,000/acre","Low (300–500mm)","NPK 60:30:20","Powdery mildew, Smut","Oct–Nov",120,"Cool-season crop; sandy loam soils with alkaline tolerance.",1635),
    # PULSES
    "blackgram":   ("pulse","🫘","Kharif / Rabi","₹22,000–₹40,000/acre","Low–Moderate","Rhizobium inoculant + NPK 20:40:20","Yellow Mosaic Virus, Cercospora","Jun / Oct",65,"Short-duration legume; fixes nitrogen; improves soil health.",7755),
    "greengram":   ("pulse","🫘","Kharif / Zaid","₹20,000–₹38,000/acre","Low–Moderate","Rhizobium + NPK 20:40:20","Yellow Mosaic, Leaf spot","Jun / Mar",60,"Quick crop (60 days); excellent for intercropping.",8558),
    "chickpea":    ("pulse","🫘","Rabi (Oct–Mar)","₹28,000–₹48,000/acre","Low (300–450mm)","Rhizobium + NPK 20:60:20","Wilt, Botrytis grey mold","Oct–Nov",100,"Cool-season legume; tolerates mild frost; black & loamy soils.",5440),
    "horsegram":   ("pulse","🌿","Kharif / Rabi","₹14,000–₹26,000/acre","Very Low","Minimal fertilizer; Rhizobium inoculant","Powdery mildew","Jun / Oct",80,"Extremely drought-tolerant; thrives in poor laterite soils.",4000),
    "lentil":      ("pulse","🍃","Rabi (Oct–Mar)","₹22,000–₹38,000/acre","Low","Rhizobium + NPK 15:40:20","Rust, Wilt","Oct–Nov",100,"Nitrogen-fixing legume; light clay to alluvial soils.",6425),
    "pigeonpeas":  ("pulse","🌿","Kharif (Jun–Oct)","₹20,000–₹38,000/acre","Low–Moderate","Rhizobium + NPK 15:30:15","Sterility mosaic, Fusarium wilt","Jun–Jul",160,"Long-duration legume; intercropped with cotton or sorghum.",7000),
    "kidneybeans": ("pulse","🫘","Kharif (Jul–Oct)","₹26,000–₹44,000/acre","Moderate","NPK 20:60:40","Anthracnose, Angular leaf spot","Jul",90,"High-value pulse; hilly regions of Uttarakhand & HP.",6000),
    "mothbeans":   ("pulse","🌱","Kharif (Jun–Sep)","₹16,000–₹28,000/acre","Very Low (<300mm)","Minimal; Rhizobium","Powdery mildew","Jun",75,"Most drought-resistant pulse; sandy desert soils of Rajasthan.",4500),
    "cowpea":      ("pulse","🌿","Kharif / Zaid","₹18,000–₹32,000/acre","Low–Moderate","Rhizobium + NPK 15:40:15","Mosaic virus, Aphids","Jun / Mar",70,"Dual-purpose (grain + fodder); warm humid conditions.",3000),
    # CASH CROPS
    "sugarcane":   ("cash","🎋","Annual (Mar–Jun plant)","₹80,000–₹1,50,000/acre","Very High (1500–2000mm)","NPK 250:80:100; Split in 3 doses","Red rot, Smut, Top shoot borer","Feb–Mar",365,"Highest water & fertilizer crop; UP produces 40% of India's sugarcane.",3150),
    "cotton":      ("cash","🌸","Kharif (May–Dec)","₹30,000–₹55,000/acre","Moderate–High","NPK 100:50:50; Boron micro-nutrient","Bollworm, Leaf curl virus","May–Jun",180,"Black cotton soils of Vidarbha are globally renowned.",6620),
    "groundnut":   ("cash","🥜","Kharif / Rabi","₹25,000–₹45,000/acre","Moderate (500–600mm)","NPK 10:40:40; Gypsum 400kg/ha at pegging","Tikka leaf spot, Root rot","Jun / Oct",120,"Sandy loam; high calcium demand; nitrogen fixer.",6377),
    "jute":        ("cash","🌿","Kharif (Mar–Jun)","₹20,000–₹35,000/acre","High (1200–1500mm)","NPK 60:30:30","Stem rot, Anthracnose","Mar–Apr",120,"Alluvial deltaic soils; requires warm humid climate.",3500),
    "tobacco":     ("cash","🍃","Rabi (Oct–Jan)","₹40,000–₹80,000/acre","Moderate","NPK 60:60:120; Low nitrogen at finish","Mosaic, Blue mold","Oct–Nov",90,"Well-drained sandy loam; Andhra Pradesh dominates India.",150),
    # OILSEEDS
    "sesame":      ("oilseed","🌱","Kharif (Jun–Aug)","₹18,000–₹32,000/acre","Low (300–400mm)","NPK 30:15:15","Phyllody, Charcoal rot","Jun–Jul",85,"Fastest maturing oilseed; sandy loam with good drainage.",15000),
    "sunflower":   ("oilseed","🌻","Rabi / Kharif","₹22,000–₹38,000/acre","Moderate (500–700mm)","NPK 80:60:60; Boron foliar spray","Alternaria, Downy mildew","Oct / Jun",90,"Photo-insensitive; can be grown in all seasons.",6400),
    "castor":      ("oilseed","🌿","Kharif (Jun–Sep)","₹20,000–₹36,000/acre","Low–Moderate","NPK 40:20:20","Semilooper, Alternaria blight","Jun–Jul",180,"Drought-tolerant; deep-rooted; sandy & loamy soils.",6485),
    "mustard":     ("oilseed","🌿","Rabi (Sep–Oct)","₹22,000–₹40,000/acre","Low–Moderate (2–3 irrigations)","NPK 60:40:30; Sulphur 20kg/ha","Alternaria, Aphids","Sep–Oct",100,"Quick-maturing winter oilseed; Rajasthan accounts for 46%.",5650),
    "soybean":     ("oilseed","🌱","Kharif (Jun–Sep)","₹25,000–₹45,000/acre","Moderate (600–700mm)","Rhizobium + NPK 20:80:40","Yellow mosaic, Girdle beetle","Jun–Jul",90,"Protein-rich; black soils of Madhya Pradesh are ideal.",4600),
    # HORTICULTURE
    "banana":      ("horticulture","🍌","Annual (Jan–Mar plant)","₹60,000–₹1,20,000/acre","Very High","NPK 200:60:300; monthly fertigation","Panama wilt, Sigatoka","Jan–Mar",300,"Highest K demand; alluvial & loamy soils; Tissue culture variety preferred.",1500),
    "mango":       ("horticulture","🥭","Summer (Mar–Jun)","₹70,000–₹1,30,000/acre","Low–Moderate","NPK 100:50:100 per tree per year","Anthracnose, Mango hopper","Perennial",1460,"Perennial; dry winters trigger flowering; Alphonso tops export value.",3000),
    "coconut":     ("plantation","🥥","Perennial","₹40,000–₹75,000/acre","High","NPK 500:320:1200g/palm/year","Bud rot, Red palm weevil","Perennial",1825,"Takes 5–7 years to bear; intercropping with cocoa/banana maximizes income.",3000),
    "tapioca":     ("horticulture","🌿","Annual (Jan–Mar)","₹30,000–₹55,000/acre","Moderate","NPK 60:60:120","Cassava mosaic, Mealybug","Jan–Mar",300,"Heavy K feeder; Kerala & Tamil Nadu primary producers.",1700),
    "papaya":      ("horticulture","🍑","Annual","₹50,000–₹90,000/acre","Moderate","NPK 200:200:250","Ringspot virus, Powdery mildew","Year-round",270,"Fast fruiting (9 months); avoid waterlogging; papain extraction adds value.",1000),
    "guava":       ("horticulture","🍐","Biannual","₹35,000–₹65,000/acre","Low–Moderate","NPK 300:200:300 per tree per year","Wilt, Fruit fly","Mar–Apr",365,"Tolerates wide soil range; high Vit-C; Allahabad Safeda is premium.",1200),
    "sapota":      ("horticulture","🟤","Biannual","₹40,000–₹70,000/acre","Moderate","NPK 600:200:600g per tree per year","Leaf webber, Sooty mould","Perennial",730,"Takes 5–6 years to fruit; long shelf life; Gujarat produces most.",800),
    "watermelon":  ("horticulture","🍉","Zaid / Kharif","₹35,000–₹70,000/acre","Moderate (400–600mm)","NPK 50:50:75 + foliar Ca-Nitrate","Mosaic, Gummy stem blight","Feb–Mar",70,"Sandy loam riverbed cultivation; Zaid crop with high market demand.",350),
    "muskmelon":   ("horticulture","🍈","Zaid (Feb–May)","₹30,000–₹55,000/acre","Low–Moderate","NPK 30:60:30","Mosaic, Powdery mildew","Feb–Mar",80,"Short-duration summer crop; sandy riverbed soils are preferred.",300),
    "grapes":      ("horticulture","🍇","Rabi (harvest Mar–May)","₹1,00,000–₹2,00,000/acre","Moderate (drip irrigation)","NPK 200:300:400; plus micronutrients","Downy mildew, Anthracnose","Perennial",730,"High-value export crop; Nashik & Sangareddy are India's grape hubs.",3000),
    "pomegranate": ("horticulture","🍎","3 crops per 2 years","₹80,000–₹1,50,000/acre","Low (drought-tolerant)","NPK 625:250:250g per tree per year","Bacterial blight, Fruit borer","Perennial",730,"High-value; semi-arid Maharashtra & Gujarat; premium export to Gulf.",5000),
    "orange":      ("horticulture","🍊","Rabi (harvest Dec–Feb)","₹60,000–₹1,10,000/acre","Moderate","NPK 400:200:400g per tree per year","Citrus canker, Tristeza virus","Perennial",1460,"Nagpur mandarin is GI-tagged; cool winters improve sugar content.",3500),
    "apple":       ("horticulture","🍏","Autumn (harvest Aug–Oct)","₹1,50,000–₹3,00,000/acre","Moderate (500–1000mm)","NPK 700:350:700g per tree per year","Scab, Fire blight","Perennial",1460,"Chilling hours (>1000hrs below 7°C) critical; Himachal Pradesh dominates.",10000),
    "pineapple":   ("horticulture","🍍","Kharif (harvest Jun–Aug)","₹40,000–₹80,000/acre","Moderate","NPK 8:2:12g per plant; ethephon for uniform flowering","Mealybug wilt, Heart rot","Perennial",540,"Assam Kew variety is globally competitive; needs acidic soils.",3000),
    # VEGETABLES
    "tomato":      ("vegetable","🍅","Rabi / Kharif","₹50,000–₹1,20,000/acre","Moderate","NPK 120:80:80; Ca-Boron foliar","Early/Late blight, Leaf curl","Oct / Jun",75,"Highest-value vegetable; hybrid varieties essential.",1000),
    "brinjal":     ("vegetable","🍆","Kharif / Rabi","₹35,000–₹70,000/acre","Moderate","NPK 100:60:60","Shoot & Fruit borer, Little leaf","Jun / Oct",70,"Year-round crop; anthracnose risk in high humidity.",600),
    "onion":       ("vegetable","🧅","Rabi (Oct–Mar)","₹40,000–₹1,00,000/acre","Moderate (6–8 irrigations)","NPK 100:50:50; Sulphur 25kg/ha","Purple blotch, Stemphylium","Oct–Nov",120,"Highly volatile market prices; Maharashtra leads India.",800),
    "potato":      ("vegetable","🥔","Rabi (Oct–Feb)","₹40,000–₹80,000/acre","High (7–10 irrigations)","NPK 180:100:150; Split urea","Late blight, Common scab","Oct–Nov",90,"Cool-season; UP produces 30% of India's potato; cold storage adds value.",1000),
    "okra":        ("vegetable","🫛","Kharif / Zaid","₹30,000–₹60,000/acre","Moderate","NPK 50:30:30","Yellow Vein Mosaic Virus","Jun / Feb",60,"Heat-tolerant; quick returns; fresh market & export both viable.",1500),
    "cabbage":     ("vegetable","🥦","Rabi (Sep–Nov)","₹25,000–₹50,000/acre","High (regular irrigation)","NPK 120:60:80; Calcium foliar","Black rot, Clubroot","Sep–Oct",90,"Cool-season; requires high nitrogen; West Bengal leads production.",600),
    "cauliflower": ("vegetable","🥦","Rabi (Sep–Jan)","₹28,000–₹55,000/acre","High","NPK 120:80:60; Boron & Molybdenum","Downy mildew, Black rot","Sep–Oct",80,"Sensitive to heat; protected cultivation extends season.",600),
    # SPICES
    "turmeric":    ("spice","💛","Kharif (Jun–Aug)","₹50,000–₹1,00,000/acre","High (regular)","NPK 90:60:120; FYM 30t/ha","Rhizome rot, Leaf blotch","Apr–May",240,"Nizamabad turmeric GI-tagged; oleoresin export growing fast.",13000),
    "chilli":      ("spice","🌶️","Kharif / Rabi","₹50,000–₹1,50,000/acre","Moderate","NPK 120:60:60; Magnesium foliar","Anthracnose, Thrips","Jun / Sep",150,"Guntur chilli (AP) is the most traded in Asia; Capsaicin extraction adds value.",20000),
    "coriander":   ("spice","🌿","Rabi (Oct–Nov)","₹18,000–₹35,000/acre","Low (3–4 irrigations)","NPK 30:20:20","Powdery mildew, Stem gall","Oct–Nov",60,"Rajasthan produces 80%; dual-use (seeds + greens); quick returns.",7500),
    "pepper":      ("spice","🫙","Perennial","₹80,000–₹1,80,000/acre","High","NPK 140:55:270g per vine per year + organic","Phytophthora, Pollu beetle","Perennial",1825,"Black gold of India; Kerala & Kodagu centers; Malabar pepper GI-tagged.",50000),
    "cardamom":    ("spice","💚","Perennial","₹1,00,000–₹3,00,000/acre","Very High (high humidity required)","NPK 75:75:150kg/ha + shade trees","Katte mosaic, Capsule rot","Perennial",1825,"Most expensive spice by weight; grown under forest canopy in Kerala.",1200),
    "ginger":      ("spice","🫚","Kharif (Apr–Jun)","₹60,000–₹1,20,000/acre","High","NPK 75:50:50 + organic mulch","Soft rot, Bacterial wilt","Apr–May",240,"India is the largest producer; Maran & Wayanad varieties are premium.",2500),
    # PLANTATION
    "tea":         ("plantation","🍵","Perennial (year-round flush)","₹80,000–₹1,50,000/acre","Very High","NPK 60:15:25 + Magnesium; frequent light doses","Blister blight, Red spider mite","Perennial",365,"Assam & Darjeeling teas are globally premium; requires acidic soils.",2000),
    "coffee":      ("plantation","☕","Perennial","₹80,000–₹1,50,000/acre","High (shade cultivation)","NPK 50:20:30 + micronutrients; foliar Mg","White stem borer, Coffee rust","Perennial",1825,"Arabica (shade-grown) & Robusta; Coorg & Chikmagalur are hubs.",8000),
    "cashew":      ("plantation","🥜","Perennial","₹35,000–₹70,000/acre","Low–Moderate","NPK 500:125:125g per tree per year","Tea mosquito bug, Stem/root rot","Perennial",1825,"Rainfed laterite slopes; kernel processing adds 5x value.",10000),
    "rubber":      ("plantation","🌲","Perennial","₹60,000–₹1,20,000/acre","Very High (>2000mm)","NPK 100:45:90kg/ha","South American Leaf Blight, Oidium","Perennial",2190,"Kerala produces 90%; latex tapping from year 6–7.",18000),
    "arecanut":    ("plantation","🌴","Perennial","₹50,000–₹1,00,000/acre","High","NPK 100:40:140g per palm per year","Koleroga, Yellow leaf disease","Perennial",1825,"Multi-storey gardens with cocoa/pepper; Karnataka dominates India.",50000),
    "cocoa":       ("plantation","🍫","Perennial","₹50,000–₹90,000/acre","High (shade needed)","NPK 100:40:120g per plant per year","Black pod rot, Capsid bug","Perennial",1825,"Under-palm cultivation in Kerala & AP; chocolate processing value chain.",2000),
    # FLOWERS
    "jasmine":     ("flower","🌸","Year-round","₹60,000–₹1,20,000/acre","Moderate","NPK 200:200:200 + FYM","Bud worm, Gall mite","Mar–Apr",365,"Tamil Nadu Mullai & Madurai Malli are world-class; daily harvest.",800),
    "marigold":    ("flower","🌼","Kharif / Rabi","₹40,000–₹80,000/acre","Moderate","NPK 120:80:80","Damping-off, Collar rot","Jun / Sep",60,"Highest-volume commercial flower; also suppresses nematodes.",300),
    "rose":        ("flower","🌹","Rabi / Year-round","₹80,000–₹2,00,000/acre","Moderate–High","NPK 200:100:150; Iron chelate","Dieback, Powdery mildew, Thrips","Sep–Oct",365,"Cut-flower export grade from Bengaluru; polyhouse premium.",1000),
}

CATEGORY_INFO = {
    "cereal":       {"label": "Cereals & Millets",    "color": "#d97706", "bg": "#fffbeb"},
    "pulse":        {"label": "Pulses",               "color": "#059669", "bg": "#ecfdf5"},
    "cash":         {"label": "Cash Crops",           "color": "#7c3aed", "bg": "#f5f3ff"},
    "oilseed":      {"label": "Oilseeds",             "color": "#ea580c", "bg": "#fff7ed"},
    "horticulture": {"label": "Horticultural Crops",  "color": "#db2777", "bg": "#fdf2f8"},
    "vegetable":    {"label": "Vegetables",           "color": "#0d9488", "bg": "#f0fdfa"},
    "spice":        {"label": "Spices & Commercial",  "color": "#dc2626", "bg": "#fef2f2"},
    "plantation":   {"label": "Plantation Crops",     "color": "#65a30d", "bg": "#f7fee7"},
    "flower":       {"label": "Flower Crops",         "color": "#9333ea", "bg": "#faf5ff"},
}

STATES = sorted([
    "Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh",
    "Goa","Gujarat","Haryana","Himachal Pradesh","Jammu & Kashmir","Jharkhand",
    "Karnataka","Kerala","Madhya Pradesh","Maharashtra","Manipur","Meghalaya",
    "Mizoram","Nagaland","Odisha","Punjab","Rajasthan","Sikkim","Tamil Nadu",
    "Telangana","Tripura","Uttar Pradesh","Uttarakhand","West Bengal",
])

SOIL_LABELS = {
    "alluvial": "Alluvial Soil 🌊",
    "black":    "Black Soil ⬛",
    "red":      "Red Soil 🔴",
    "laterite": "Laterite Soil 🟠",
    "sandy":    "Sandy Soil 🏜️",
    "loamy":    "Loamy Soil 🟫",
    "clay":     "Clay Soil 🧱",
}
SOIL_CHARS = {
    "alluvial": "Fertile; good water retention; found in river plains — ideal for most crops",
    "black":    "High clay content; moisture-retentive; ideal for cotton & dryland crops",
    "red":      "Well-drained; iron-rich; good for millets, pulses & horticulture",
    "laterite": "Acidic; good drainage; suits plantation, spice & coffee crops",
    "sandy":    "Low water retention; fast-draining; needs frequent light irrigation",
    "loamy":    "Best agricultural soil; balanced texture & nutrients; suits all crops",
    "clay":     "High water retention; prone to waterlogging; needs land drainage",
}

SEASONS = {
    "kharif": "Kharif ☔ (Jun–Nov, Monsoon season)",
    "rabi":   "Rabi ❄️ (Oct–Mar, Winter/cool season)",
    "zaid":   "Zaid ☀️ (Mar–Jun, Hot summer season)",
}
WATER_OPTS = {
    "rainfed":   "Rainfed 🌧️ (monsoon dependent)",
    "irrigated": "Irrigated 💧 (assured water supply)",
}

CLIMATE_ZONES = {
    "Andhra Pradesh":"Tropical","Arunachal Pradesh":"Subtropical Humid",
    "Assam":"Subtropical Humid","Bihar":"Subtropical","Chhattisgarh":"Subtropical",
    "Goa":"Tropical Wet","Gujarat":"Semi-Arid","Haryana":"Semi-Arid",
    "Himachal Pradesh":"Temperate","Jammu & Kashmir":"Temperate Alpine",
    "Jharkhand":"Subtropical","Karnataka":"Tropical Semi-Arid",
    "Kerala":"Tropical Humid","Madhya Pradesh":"Subtropical Semi-Arid",
    "Maharashtra":"Semi-Arid Tropical","Manipur":"Subtropical Humid",
    "Meghalaya":"Subtropical Humid","Mizoram":"Subtropical Humid",
    "Nagaland":"Subtropical Humid","Odisha":"Tropical",
    "Punjab":"Semi-Arid","Rajasthan":"Arid","Sikkim":"Temperate",
    "Tamil Nadu":"Tropical","Telangana":"Semi-Arid Tropical",
    "Tripura":"Subtropical Humid","Uttar Pradesh":"Subtropical",
    "Uttarakhand":"Temperate","West Bengal":"Tropical Humid",
}
DROUGHT_RISK = {
    "Rajasthan":"High","Gujarat":"Moderate","Maharashtra":"Moderate",
    "Andhra Pradesh":"Moderate","Telangana":"Moderate","Karnataka":"Moderate",
    "Haryana":"Low–Moderate","Punjab":"Low","Uttar Pradesh":"Low",
    "Madhya Pradesh":"Moderate","Odisha":"Low–Moderate",
    "West Bengal":"Low","Assam":"Low","Kerala":"Low","Tamil Nadu":"Low–Moderate",
    "Bihar":"Low","Jharkhand":"Moderate","Chhattisgarh":"Low–Moderate",
    "Himachal Pradesh":"Low","Uttarakhand":"Low","Jammu & Kashmir":"Low",
    "Arunachal Pradesh":"Low","Meghalaya":"Low","Manipur":"Low",
    "Mizoram":"Low","Nagaland":"Low","Tripura":"Low","Sikkim":"Low","Goa":"Low",
}
FLOOD_RISK = {
    "Assam":"High","Bihar":"High","West Bengal":"High","Odisha":"High",
    "Uttar Pradesh":"Moderate","Andhra Pradesh":"Moderate","Kerala":"Moderate",
    "Maharashtra":"Moderate","Rajasthan":"Low","Gujarat":"Low",
    "Punjab":"Low","Haryana":"Low","Karnataka":"Low","Tamil Nadu":"Low–Moderate",
    "Telangana":"Low","Madhya Pradesh":"Low","Chhattisgarh":"Low",
    "Jharkhand":"Low","Himachal Pradesh":"Low","Uttarakhand":"Low",
    "Jammu & Kashmir":"Low","Arunachal Pradesh":"Moderate",
    "Meghalaya":"Moderate","Manipur":"Low","Mizoram":"Low",
    "Nagaland":"Low","Tripura":"Moderate","Sikkim":"Low","Goa":"Low",
}


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading AI ensemble model...")
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    st.error("Model file not found. Please run the training script first.")
    st.stop()

bundle = load_model()


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def predict_top_crops(n, p, k, temp, humidity, ph, rainfall,
                      soil_type, season, water_avail, top_k=5):
    soil_enc   = bundle["soil_map"].get(soil_type, 3)
    season_enc = bundle["season_map"].get(season, 1)
    water_enc  = bundle["water_map"].get(water_avail, 0)

    X = np.array([[n, p, k, temp, humidity, ph, rainfall,
                   soil_enc, season_enc, water_enc]])

    proba   = bundle["model"].predict_proba(X)[0]
    le      = bundle["le_crop"]
    top_idx = np.argsort(proba)[::-1][:top_k]

    results = []
    for idx in top_idx:
        crop_name = le.inverse_transform([idx])[0]
        conf      = round(float(proba[idx]) * 100, 1)
        info      = CROP_DB.get(crop_name)
        if info:
            results.append((crop_name, conf, info))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# WEATHER API
# ══════════════════════════════════════════════════════════════════════════════
def fetch_weather(city: str) -> dict:
    if OWM_API_KEY is None:
        raise ValueError("OWM_API_KEY not configured in st.secrets.")
    if not city.strip():
        raise ValueError("Please enter a city name.")
    try:
        resp = requests.get(OWM_URL,
            params={"q": city.strip(), "appid": OWM_API_KEY, "units": "metric"},
            timeout=6)
    except requests.exceptions.ConnectionError:
        raise ValueError("No internet connection.")
    except requests.exceptions.Timeout:
        raise ValueError("Weather API timed out.")
    if resp.status_code == 401: raise ValueError("Invalid API key.")
    if resp.status_code == 404: raise ValueError(f"City '{city}' not found.")
    if resp.status_code != 200: raise ValueError(f"API error HTTP {resp.status_code}.")
    d = resp.json()
    return {
        "temperature": round(d["main"]["temp"], 1),
        "humidity":    round(d["main"]["humidity"], 1),
        "description": d["weather"][0]["description"].title(),
        "city_name":   d["name"] + ", " + d["sys"]["country"],
        "wind_speed":  round(d["wind"]["speed"] * 3.6, 1),
        "pressure":    d["main"]["pressure"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# PDF REPORT
# ══════════════════════════════════════════════════════════════════════════════
def generate_pdf(inputs: dict, results: list, state_name: str, accuracy: float) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    GREEN = colors.HexColor("#14532d")
    LG    = colors.HexColor("#dcfce7")
    GREY  = colors.HexColor("#374151")

    def ps(name, **kw):
        return ParagraphStyle(name, parent=styles["Normal"], **kw)

    TI = ps("TI", textColor=GREEN, fontSize=18, fontName="Helvetica-Bold", spaceAfter=3)
    SB = ps("SB", textColor=colors.HexColor("#6b7280"), fontSize=8, spaceAfter=2)
    SH = ps("SH", textColor=GREEN, fontSize=11, fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=5)
    BD = ps("BD", textColor=GREY, fontSize=9, leading=14)
    CR = ps("CR", textColor=GREEN, fontSize=16, fontName="Helvetica-Bold", alignment=1)
    NT = ps("NT", textColor=colors.HexColor("#9ca3af"), fontSize=7, leading=11)

    def hr():
        return HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#bbf7d0"), spaceAfter=5)

    story = []
    now = datetime.datetime.now().strftime("%d %B %Y, %I:%M %p")

    story += [
        Paragraph("🌾 Smart Crop Advisor — India Edition", TI),
        Paragraph(f"Generated: {now}  |  State: {state_name}  |  Ensemble Model Accuracy: {accuracy*100:.1f}%", SB),
        Spacer(1, 0.15*cm), hr()
    ]

    # Top-5 table
    story += [Paragraph("Top 5 Crop Recommendations", SH)]
    rows = [["Rank", "Crop", "Category", "Confidence", "Season", "Est. Profit/Acre"]]
    for i, (crop, conf, info) in enumerate(results, 1):
        cat_label = CATEGORY_INFO.get(info[0], {}).get("label", info[0])
        rows.append([str(i), crop.title(), cat_label, f"{conf}%", info[2], info[3]])
    t = Table(rows, colWidths=[1*cm, 3.2*cm, 3.5*cm, 2*cm, 3.8*cm, 4.5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),GREEN),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,LG]),
        ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#86efac")),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),5),
    ]))
    story += [t, Spacer(1, 0.3*cm), hr()]

    # Primary crop detail
    top_crop, top_conf, top_info = results[0]
    story += [Paragraph(f"{top_info[1]} Primary Pick: {top_crop.title()} ({top_conf}% confidence)", CR), Spacer(1, 0.15*cm)]

    detail = [
        ["Field", "Detail"],
        ["Category",          CATEGORY_INFO.get(top_info[0],{}).get("label", top_info[0])],
        ["Growing Season",    top_info[2]],
        ["Estimated Profit",  top_info[3]],
        ["Water Requirement", top_info[4]],
        ["Fertilizer Guide",  top_info[5]],
        ["Disease Risks",     top_info[6]],
        ["Best Sowing Time",  top_info[7]],
        ["Harvest Duration",  f"{top_info[8]} days"],
        ["MSP (approx.)",     f"Rs.{top_info[10]}/quintal"],
    ]
    dt = Table(detail, colWidths=[4.5*cm, 13.5*cm])
    dt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),GREEN),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,LG]),
        ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#86efac")),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),5),("VALIGN",(0,0),(-1,-1),"TOP"),
    ]))
    story += [dt, Spacer(1, 0.25*cm)]
    story += [Paragraph("Agronomist Insight", SH), Paragraph(top_info[9], BD), Spacer(1,0.25*cm), hr()]

    # Climate & Risk analysis
    story += [Paragraph(f"Regional Analysis — {state_name}", SH)]
    clim  = CLIMATE_ZONES.get(state_name, "Tropical")
    dris  = DROUGHT_RISK.get(state_name, "Low")
    fris  = FLOOD_RISK.get(state_name, "Low")
    risk_data = [
        ["Parameter", "Value"],
        ["Climate Zone", clim],
        ["Drought Risk", dris],
        ["Flood Risk", fris],
        ["Soil Type Used", SOIL_LABELS[inputs["soil_type"]]],
        ["Season", SEASONS[inputs["season"]]],
        ["Water Availability", WATER_OPTS[inputs["water_avail"]]],
    ]
    rt = Table(risk_data, colWidths=[6*cm, 12*cm])
    rt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),GREEN),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,LG]),
        ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#86efac")),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),5),
    ]))
    story += [rt, Spacer(1, 0.25*cm), hr()]

    # Input parameters
    story += [Paragraph("Field Input Parameters", SH)]
    ip = [
        ["Parameter","Value","Unit"],
        ["Nitrogen (N)", str(inputs["N"]), "kg/ha"],
        ["Phosphorus (P)", str(inputs["P"]), "kg/ha"],
        ["Potassium (K)", str(inputs["K"]), "kg/ha"],
        ["Temperature", str(inputs["temperature"]), "Celsius"],
        ["Humidity", str(inputs["humidity"]), "%"],
        ["pH Value", str(inputs["ph"]), "—"],
        ["Rainfall", str(inputs["rainfall"]), "mm"],
    ]
    it = Table(ip, colWidths=[5.5*cm, 5*cm, 7.5*cm])
    it.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),GREEN),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,LG]),
        ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#86efac")),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),5),
    ]))
    story += [it, Spacer(1,0.5*cm), hr()]
    story += [Paragraph(
        "Disclaimer: Generated by an AI ensemble model (RF+XGBoost+DT) trained on 4,960 agronomic records. "
        "Always verify with your local KVK (Krishi Vigyan Kendra) before final crop decisions. "
        "MSP values are indicative and subject to annual Government revision.",
        NT)]

    doc.build(story)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart Crop Advisor — India",
    page_icon="🌾", layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background: #f8faf8; }

.hero {
  background: linear-gradient(135deg, #14532d 0%, #15803d 45%, #4ade80 100%);
  border-radius: 20px; padding: 2.5rem 2.5rem 2rem; color: white;
  margin-bottom: 1.6rem; box-shadow: 0 12px 40px rgba(20,83,45,0.3);
  position: relative; overflow: hidden;
}
.hero::before { content:"🌾"; font-size:8rem; opacity:0.08; position:absolute; right:2rem; top:-1rem; }
.hero h1 { font-family:'Sora',sans-serif; font-size:2rem; font-weight:800; margin:0; letter-spacing:-0.5px; }
.hero p  { font-size:0.95rem; opacity:0.9; margin-top:0.4rem; }
.badge   { display:inline-block; background:rgba(255,255,255,0.2); border:1px solid rgba(255,255,255,0.3);
           border-radius:999px; padding:0.2rem 0.8rem; font-size:0.78rem; font-weight:600;
           margin-right:0.4rem; margin-top:0.5rem; }

.section-hd {
  font-family:'Sora',sans-serif; font-size:0.82rem; font-weight:700; color:#15803d;
  text-transform:uppercase; letter-spacing:0.08em; border-left:3px solid #4ade80;
  padding-left:0.6rem; margin-bottom:0.7rem;
}

.wx-card { background:linear-gradient(135deg,#075985,#0ea5e9); border-radius:14px;
           padding:0.9rem 1.3rem; color:white; margin-bottom:0.9rem;
           box-shadow:0 4px 16px rgba(7,89,133,0.25); }
.wx-city { font-size:0.72rem; opacity:0.85; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; }
.wx-row  { display:flex; gap:1.5rem; margin-top:0.5rem; flex-wrap:wrap; }
.wx-item { text-align:center; }
.wx-num  { font-family:'Sora',sans-serif; font-size:1.4rem; font-weight:700; }
.wx-lbl  { font-size:0.7rem; opacity:0.82; }

.rank1 { background:linear-gradient(135deg,#14532d,#16a34a); color:white; border-radius:16px;
         padding:1.5rem; text-align:center; margin-bottom:0.9rem;
         box-shadow:0 6px 24px rgba(20,83,45,0.25); }
.rank1 .emo  { font-size:3.2rem; }
.rank1 .name { font-family:'Sora',sans-serif; font-size:1.7rem; font-weight:800;
               text-transform:capitalize; margin:0.25rem 0; }
.rank1 .conf { background:rgba(255,255,255,0.2); border-radius:999px; display:inline-block;
               padding:0.18rem 0.75rem; font-size:0.82rem; font-weight:600; }

.ig { display:grid; grid-template-columns:1fr 1fr; gap:0.55rem; margin-bottom:0.8rem; }
.ig-i { background:#f8faf8; border:1px solid #e5e7eb; border-radius:10px; padding:0.65rem 0.85rem; }
.ig-l { font-size:0.68rem; color:#6b7280; text-transform:uppercase; letter-spacing:0.07em; font-weight:600; }
.ig-v { font-size:0.88rem; font-weight:700; color:#15803d; margin-top:0.12rem; }

.alt { background:white; border:1px solid #d1fae5; border-radius:12px; padding:0.8rem 1rem;
       margin-bottom:0.55rem; display:flex; align-items:center; gap:0.75rem; }
.alt-rk { background:#f0fdf4; color:#15803d; font-weight:700; font-size:0.82rem;
          width:26px; height:26px; border-radius:50%; display:flex; align-items:center;
          justify-content:center; flex-shrink:0; }
.alt-em { font-size:1.4rem; flex-shrink:0; }
.alt-nm { font-weight:700; color:#111827; text-transform:capitalize; font-size:0.9rem; }
.alt-mt { font-size:0.76rem; color:#6b7280; }
.alt-cf { margin-left:auto; background:#dcfce7; color:#15803d; border-radius:999px;
          padding:0.18rem 0.65rem; font-size:0.78rem; font-weight:600; flex-shrink:0; }

.al-w { background:#fffbeb; border:1px solid #fde68a; color:#92400e; border-radius:10px; padding:0.7rem 0.9rem; margin-bottom:0.5rem; font-size:0.85rem; }
.al-i { background:#eff6ff; border:1px solid #bfdbfe; color:#1d4ed8; border-radius:10px; padding:0.7rem 0.9rem; margin-bottom:0.5rem; font-size:0.85rem; }
.al-g { background:#f0fdf4; border:1px solid #bbf7d0; color:#14532d; border-radius:10px; padding:0.7rem 0.9rem; margin-bottom:0.5rem; font-size:0.85rem; }

div.stButton > button {
  background:linear-gradient(135deg,#14532d,#16a34a); color:white; border:none;
  border-radius:10px; padding:0.7rem 2rem; font-size:0.95rem; font-weight:600;
  width:100%; box-shadow:0 4px 15px rgba(20,83,45,0.3); transition:all 0.2s;
}
div.stButton > button:hover { filter:brightness(1.08); transform:translateY(-1px); }
.stSlider label { font-weight:600; color:#374151; font-size:0.88rem; }
#MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("wx", None), ("wx_err", ""), ("result", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

acc     = bundle["accuracy"]
n_crops = bundle["n_crops"]

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <h1>🌾 Smart Crop Advisor — India Edition</h1>
  <p>AI-powered precision agriculture for all Indian states · Ensemble ML (RF + XGBoost + DT)</p>
  <span class="badge">📊 Accuracy: {acc*100:.1f}%</span>
  <span class="badge">🌱 {n_crops} Crops</span>
  <span class="badge">🗺️ 29 States</span>
  <span class="badge">🧪 10 Input Parameters</span>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.05, 0.95], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT PANEL
# ══════════════════════════════════════════════════════════════════════════════
with left:

    # Location & state
    st.markdown('<p class="section-hd">📍 Location & State</p>', unsafe_allow_html=True)
    state_sel = st.selectbox("Select Your State", STATES,
                             index=STATES.index("Tamil Nadu"))

    cz   = CLIMATE_ZONES.get(state_sel, "Tropical")
    dr   = DROUGHT_RISK.get(state_sel, "Low")
    fr   = FLOOD_RISK.get(state_sel, "Low")

    dr_col = "#dc2626" if "High" in dr else "#d97706" if "Moderate" in dr else "#16a34a"
    fr_col = "#dc2626" if "High" in fr else "#d97706" if "Moderate" in fr else "#16a34a"

    st.markdown(f"""
    <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:0.9rem;">
      <span style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:0.28rem 0.65rem;font-size:0.76rem;color:#14532d;font-weight:600;">🌍 {cz}</span>
      <span style="background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;padding:0.28rem 0.65rem;font-size:0.76rem;color:{dr_col};font-weight:600;">🏜️ Drought: {dr}</span>
      <span style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:0.28rem 0.65rem;font-size:0.76rem;color:{fr_col};font-weight:600;">🌊 Flood: {fr}</span>
    </div>
    """, unsafe_allow_html=True)

    # Weather
    st.markdown('<p class="section-hd">🌤️ Live Weather Auto-Fill</p>', unsafe_allow_html=True)
    if OWM_API_KEY:
        cc, bc = st.columns([3, 1.2])
        with cc:
            city_in = st.text_input("Enter city name", placeholder="e.g. Chennai, Pune, Jaipur...",
                                    label_visibility="collapsed")
        with bc:
            if st.button("🌤 Fetch"):
                try:
                    st.session_state.wx     = fetch_weather(city_in)
                    st.session_state.wx_err = ""
                except ValueError as e:
                    st.session_state.wx_err = str(e)
                    st.session_state.wx     = None
        if st.session_state.wx_err:
            st.error(f"⚠️ {st.session_state.wx_err}")
        if st.session_state.wx:
            wx = st.session_state.wx
            st.markdown(f"""
            <div class="wx-card">
              <div class="wx-city">📍 {wx['city_name']} · {wx['description']}</div>
              <div class="wx-row">
                <div class="wx-item"><div class="wx-num">{wx['temperature']}°C</div><div class="wx-lbl">Temperature</div></div>
                <div class="wx-item"><div class="wx-num">{wx['humidity']}%</div><div class="wx-lbl">Humidity</div></div>
                <div class="wx-item"><div class="wx-num">{wx['wind_speed']} km/h</div><div class="wx-lbl">Wind Speed</div></div>
                <div class="wx-item"><div class="wx-num">{wx['pressure']} hPa</div><div class="wx-lbl">Pressure</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.caption("✅ Temperature & Humidity sliders pre-filled. Adjust if needed.")
    else:
        st.info("🔑 Add OWM_API_KEY to `.streamlit/secrets.toml` to enable live weather auto-fill.", icon="ℹ️")

    st.markdown("---")

    # Season & Water
    st.markdown('<p class="section-hd">🌦️ Season & Water Availability</p>', unsafe_allow_html=True)
    sa, wa = st.columns(2)
    with sa:
        season_key = st.selectbox("Season", list(SEASONS.keys()),
                                  format_func=lambda k: SEASONS[k])
    with wa:
        water_key = st.selectbox("Water Source", list(WATER_OPTS.keys()),
                                 format_func=lambda k: WATER_OPTS[k])

    # Soil
    st.markdown('<p class="section-hd">🪨 Soil Type</p>', unsafe_allow_html=True)
    soil_key = st.selectbox("Soil Type", list(SOIL_LABELS.keys()),
                            format_func=lambda k: SOIL_LABELS[k])
    st.caption(f"ℹ️ {SOIL_CHARS[soil_key]}")
    st.markdown("---")

    # Nutrients
    st.markdown('<p class="section-hd">🧪 Soil Nutrient Analysis (kg/ha)</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: N = st.slider("Nitrogen (N)",   0, 200, 60)
    with c2: P = st.slider("Phosphorus (P)", 5, 150, 50)
    with c3: K = st.slider("Potassium (K)",  5, 210, 50)

    st.markdown('<p class="section-hd" style="margin-top:1rem;">🌡️ Climate Conditions</p>', unsafe_allow_html=True)
    wx      = st.session_state.wx
    t_def   = max(5.0, min(45.0, float(wx["temperature"]) if wx else 28.0))
    h_def   = max(10.0, min(100.0, float(wx["humidity"]) if wx else 65.0))

    c4, c5 = st.columns(2)
    with c4:
        temperature = st.slider("Temperature (°C)", 5.0, 45.0, t_def, 0.1)
        humidity    = st.slider("Humidity (%)",    10.0, 100.0, h_def, 0.1)
    with c5:
        ph       = st.slider("pH Value",      3.5, 9.9,  6.5,  0.01)
        rainfall = st.slider("Rainfall (mm)", 20.0, 400.0, 120.0, 1.0)

    st.markdown("")
    predict_btn = st.button("🔍 Get AI Crop Recommendations", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL
# ══════════════════════════════════════════════════════════════════════════════
with right:
    st.markdown('<p class="section-hd">📊 AI Recommendations</p>', unsafe_allow_html=True)

    if predict_btn:
        with st.spinner("Analysing field conditions across 62 crops..."):
            results = predict_top_crops(N, P, K, temperature, humidity, ph, rainfall,
                                        soil_key, season_key, water_key, top_k=5)
        st.session_state.result = {
            "results": results,
            "inputs":  dict(N=N, P=P, K=K, temperature=temperature,
                            humidity=humidity, ph=ph, rainfall=rainfall,
                            soil_type=soil_key, season=season_key,
                            water_avail=water_key),
            "state":   state_sel,
        }

    R = st.session_state.result
    if R:
        res        = R["results"]
        inp        = R["inputs"]
        state_name = R["state"]

        if not res:
            st.warning("No matching crops found. Try adjusting your parameters.")
        else:
            top_crop, top_conf, top_info = res[0]
            cat_info = CATEGORY_INFO.get(top_info[0], {})

            # Primary card
            st.markdown(f"""
            <div class="rank1">
              <div class="emo">{top_info[1]}</div>
              <div class="name">{top_crop.title()}</div>
              <div class="conf">🎯 Confidence: {top_conf}%</div>
              <div style="margin-top:0.4rem;font-size:0.82rem;opacity:0.85;">{cat_info.get('label','')}</div>
            </div>
            """, unsafe_allow_html=True)

            # Info grid
            msp_val = f"₹{top_info[10]}/qtl" if len(top_info) > 10 else "Market-linked"
            st.markdown(f"""
            <div class="ig">
              <div class="ig-i"><div class="ig-l">💰 Est. Profit</div><div class="ig-v">{top_info[3]}</div></div>
              <div class="ig-i"><div class="ig-l">📅 Growing Season</div><div class="ig-v">{top_info[2]}</div></div>
              <div class="ig-i"><div class="ig-l">💧 Water Requirement</div><div class="ig-v">{top_info[4]}</div></div>
              <div class="ig-i"><div class="ig-l">⏱️ Harvest Duration</div><div class="ig-v">{top_info[8]} days</div></div>
              <div class="ig-i"><div class="ig-l">🌱 Best Sowing Time</div><div class="ig-v">{top_info[7]}</div></div>
              <div class="ig-i"><div class="ig-l">📈 MSP (Approx.)</div><div class="ig-v">{msp_val}</div></div>
            </div>
            """, unsafe_allow_html=True)

            # Alerts
            d_risk = DROUGHT_RISK.get(state_name, "Low")
            f_risk = FLOOD_RISK.get(state_name, "Low")
            if "High" in d_risk:
                st.markdown(f'<div class="al-w">⚠️ <strong>Drought Alert:</strong> {state_name} faces HIGH drought risk. Prefer drought-tolerant varieties and adopt drip/micro-irrigation.</div>', unsafe_allow_html=True)
            if "High" in f_risk:
                st.markdown(f'<div class="al-i">🌊 <strong>Flood Alert:</strong> {state_name} has HIGH flood risk. Ensure field drainage and raised-bed cultivation.</div>', unsafe_allow_html=True)
            if ph < 5.5:
                st.markdown('<div class="al-w">⚠️ <strong>Low pH (Acidic):</strong> Apply agricultural lime @ 2–4 t/ha to raise pH before sowing.</div>', unsafe_allow_html=True)
            elif ph > 8.5:
                st.markdown('<div class="al-w">⚠️ <strong>High pH (Alkaline):</strong> Apply gypsum or elemental sulphur to reduce pH levels.</div>', unsafe_allow_html=True)
            if N < 20:
                st.markdown('<div class="al-w">⚠️ <strong>Low Nitrogen:</strong> Apply basal dose of Urea (45 kg/ha) or incorporate green manure before sowing.</div>', unsafe_allow_html=True)
            if water_key == "rainfed" and rainfall < 50:
                st.markdown('<div class="al-w">⚠️ <strong>Low Rainfall + Rainfed:</strong> Consider drought-tolerant crops like Bajra, Jowar, or Castor.</div>', unsafe_allow_html=True)

            st.markdown(f'<div class="al-g">🌿 <strong>Fertilizer Guide:</strong> {top_info[5]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="al-w">🦠 <strong>Disease Risk:</strong> {top_info[6]} — consult your local KVK for preventive spray schedule.</div>', unsafe_allow_html=True)

            # Insight
            with st.expander("🤖 Agronomist's Insight", expanded=True):
                st.write(top_info[9])

            # Alternatives
            st.markdown('<p class="section-hd" style="margin-top:1rem;">🌿 Alternative Crops (Top Picks)</p>', unsafe_allow_html=True)
            for rank, (crop, conf, info) in enumerate(res[1:], 2):
                cinfo = CATEGORY_INFO.get(info[0], {})
                st.markdown(f"""
                <div class="alt">
                  <div class="alt-rk">{rank}</div>
                  <div class="alt-em">{info[1]}</div>
                  <div>
                    <div class="alt-nm">{crop.title()}</div>
                    <div class="alt-mt">{cinfo.get('label','')} · {info[2]} · {info[3]}</div>
                  </div>
                  <div class="alt-cf">{conf}%</div>
                </div>
                """, unsafe_allow_html=True)

            # Input summary
            with st.expander("📋 Full Input Summary"):
                summary_df = pd.DataFrame({
                    "Parameter": ["Nitrogen","Phosphorus","Potassium","Temperature",
                                  "Humidity","pH","Rainfall","Soil Type","Season",
                                  "Water Availability","State"],
                    "Value": [f"{inp['N']} kg/ha", f"{inp['P']} kg/ha",
                              f"{inp['K']} kg/ha", f"{inp['temperature']}°C",
                              f"{inp['humidity']}%", inp['ph'],
                              f"{inp['rainfall']} mm", SOIL_LABELS[inp['soil_type']],
                              SEASONS[inp['season']], WATER_OPTS[inp['water_avail']],
                              state_name]
                })
                st.table(summary_df.set_index("Parameter"))

            # PDF download
            pdf_bytes = generate_pdf(inp, res, state_name, acc)
            st.download_button(
                "📄 Download Full Report (PDF)", pdf_bytes,
                file_name=f"crop_report_{top_crop}_{state_name.replace(' ','_')}.pdf",
                mime="application/pdf", use_container_width=True
            )
    else:
        st.markdown("""
        <div style="background:#f0fdf4;border:2px dashed #86efac;border-radius:16px;
                    padding:3rem 2rem;text-align:center;color:#6b7280;">
          <div style="font-size:3rem;margin-bottom:0.8rem;">🌿</div>
          <p style="font-size:1rem;font-weight:600;color:#374151;">
            Configure your field on the left<br>then click <em>Get AI Crop Recommendations</em>.
          </p>
          <p style="font-size:0.83rem;margin-top:0.5rem;">
            Top-5 recommendations with confidence scores,<br>
            regional alerts, fertilizer guide & PDF report.
          </p>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style="text-align:center;color:#9ca3af;font-size:0.78rem;">
  🌾 Smart Crop Advisor — India v3 &nbsp;·&nbsp;
  Ensemble ML: Random Forest + XGBoost + Decision Tree &nbsp;·&nbsp;
  62 Crops · All Indian States &nbsp;·&nbsp; Built with Python & Streamlit
</p>
""", unsafe_allow_html=True)
