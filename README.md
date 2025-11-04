# 🇲🇾 Malaysia Used Car Depreciation Dashboard

A fully interactive **Dash web application** analyzing **used car depreciation trends** across Malaysia.  
This project explores how price, age, mileage, and location affect resale value using over **50,000 cleaned car listings** from *Mudah.my*.

---

## 📊 Overview

The dashboard provides a visual, data-driven look into the Malaysian used car market:
- Identify **depreciation curves** by brand and model.  
- Compare **price drop rates** (% loss from original value).  
- Analyze **regional differences** in listing price, mileage, and age.  
- Explore **state-level metrics** and brand-level summaries.  

Built with **Plotly Dash**, **Pandas**, and **Plotly Express**, the app allows smooth filtering and comparison across brands, models, and locations.

---

## 🧩 Key Features

### 🔹 Depreciation Analysis
- Interactive **scatter + exponential fit** curves for each model.  
- Displays estimated **initial value**, **time to −20%**, and **time to −50% depreciation**.  
- Color-coded by mileage to show variation.

### 🔹 Comparison View
- Compare up to **two models** side by side.  
- Displays **percent drop vs. vehicle age** for clearer cross-model comparison.

### 🔹 Contour Map (Price Surface)
- Smoothed **contour plot** showing average price as a function of **age** and **mileage**.  
- Highlights how wear and age interact to influence pricing.

### 🔹 State-Level Market Insights
- Table, bar chart, and choropleth map showing **median price, age, and mileage** per state.  
- Fully interactive — click states or bars to filter across views.  
- KPIs summarize total listings, median price, median age, and mileage.

---

## ⚙️ Data & Methodology

- **Source:** Scraped from [Mudah.my](https://www.mudah.my) (public used car listings).  
- **Scope:** Top 10 brands by listing volume, with representative models across segments (A/B/C/D, SUV, truck).  
- **Cleaning:** Outliers removed, standard units enforced, models with <100 listings excluded.  
- **Final dataset:** ~50,000 listings across 50+ models.  
- **Metrics:** Median price, age, and mileage for robustness.

---

## 🚀 Tech Stack

| Layer | Tools |
|-------|-------|
| **Frontend** | Plotly Dash, Dash Bootstrap Components |
| **Backend** | Flask (via Dash) |
| **Data** | Pandas, NumPy, SciPy |
| **Visualization** | Plotly Express, Mapbox |
| **Deployment** | Gunicorn + Sevalla App Hosting ($5 plan) |
| **Version Control** | Git & GitHub |

---

## 🧠 Insights (Example Findings)
- Malaysian brands (e.g., **Perodua**) and Japanese models (**Honda, Toyota**) retain value better than continental cars.  
- **Age** tends to drive price loss more strongly than **mileage** after ~5 years.  
- **Urban states** (Selangor, KL, Penang) show higher median prices due to demand and newer stock.  

---

## 💻 Local Setup & Running the App

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git

2. **Create a virtual environment**
   ```bash
   python -m venv dashboard_env
   source dashboard_env/bin/activate   # on macOS/Linux
   dashboard_env\Scripts\activate      # on Windows
   
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   
4. **Run the app**
   ```bash
   python app.py

Then open http://127.0.0.1:8050 in your browser.
