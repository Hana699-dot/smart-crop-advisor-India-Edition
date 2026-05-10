# 🌾 Smart Crop Advisor — India Edition (v3)

AI-powered precision agriculture system for all Indian states.
Ensemble ML (Random Forest + XGBoost + Decision Tree) · 62 crops · 10 input parameters.

> **Security:** API key loaded via st.secrets — never hardcoded. Safe for public GitHub.

---

## 📁 Project Structure

```
smart_crop_project/
├── app.py                        # Main Streamlit application
├── india_crop_model.pkl          # Trained ensemble model bundle
├── india_crop_dataset.csv        # 4,960-record agronomic dataset
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── .gitignore                    # Excludes secrets.toml
├── .streamlit/
│   └── secrets.toml              # LOCAL ONLY — never commit!
└── (legacy CSVs for reference)
```

---

## 🌱 Crops Covered (62)

| Category | Crops |
|---|---|
| Cereals & Millets | Rice, Wheat, Maize, Jowar, Bajra, Ragi, Barley |
| Pulses | Blackgram, Greengram, Chickpea, Horsegram, Lentil, Pigeonpeas, Kidneybeans, Mothbeans, Cowpea |
| Cash Crops | Sugarcane, Cotton, Groundnut, Jute, Tobacco |
| Oilseeds | Sesame, Sunflower, Castor, Mustard, Soybean |
| Horticultural | Banana, Mango, Tapioca, Papaya, Guava, Sapota, Watermelon, Muskmelon, Grapes, Pomegranate, Orange, Apple, Pineapple |
| Vegetables | Tomato, Brinjal, Onion, Potato, Okra, Cabbage, Cauliflower |
| Spices | Turmeric, Chilli, Coriander, Pepper, Cardamom, Ginger |
| Plantation | Coconut, Tea, Coffee, Cashew, Rubber, Arecanut, Cocoa |
| Flowers | Jasmine, Marigold, Rose |

---

## 🤖 ML Architecture

```
Ensemble Voting Classifier (soft voting)
├── Random Forest  (300 trees)
├── XGBoost        (300 estimators, lr=0.1, depth=8)
└── Decision Tree  (max_depth=25)

Training: 4,960 records · 62 crops · 10 features · 80/20 stratified split
Top-1 accuracy: ~65%  |  Top-3: ~90%  |  Top-5: ~95%
```

---

## 🚀 Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/smart_crop_project.git
cd smart_crop_project
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Edit .streamlit/secrets.toml → add your OWM API key
streamlit run app.py
```

---

## ☁️ Streamlit Cloud Deployment

1. Push to GitHub (secrets.toml is gitignored automatically)
2. Go to share.streamlit.io → New app → select repo → main file: app.py
3. Settings → Secrets → paste:
   ```toml
   OWM_API_KEY = "your_actual_key_here"
   ```
4. Save — app restarts with key loaded securely

---

## 📊 Features

- Top-5 crop recommendations with confidence scores
- Regional climate zone, drought & flood risk analysis per state
- 7 soil types with agronomic guidance
- Kharif / Rabi / Zaid season awareness
- Live weather auto-fill via OpenWeatherMap API
- Smart alerts: pH, nitrogen, drought, flood warnings
- Fertilizer recommendations & disease risk per crop
- MSP indicative values & harvest duration
- Downloadable PDF field report

## 🛠️ Stack

Streamlit · Scikit-learn · XGBoost · Pandas · NumPy · ReportLab · OpenWeatherMap API
