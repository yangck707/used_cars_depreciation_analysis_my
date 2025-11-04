import json
import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc

from dash.dash_table import DataTable, FormatTemplate
from dash.dash_table.Format import Format, Group

# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# LOADING DATA
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
LISTINGS_CSV = "data/listings.csv"  
GEOJSON_PATH = "data/geoBoundaries-MYS-ADM1_simplified.geojson"
TOP10_CSV = "data/carlistings_region.csv"

# Load listings
df = pd.read_csv(LISTINGS_CSV)
df = df.dropna(subset=["state", "price", "age"])
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["age"] = pd.to_numeric(df["age"], errors="coerce")
if "mileage" in df.columns:
    df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce")
else:
    df["mileage"] = np.nan

# Uniform strings
for c in ["state", "brand", "model"]:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip()

# Load GeoJSON
with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
    MYS = json.load(f)
FEATURE_ID_KEY = "properties.shapeName"




# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# SECTION 2: Market Coverage — Top Brands & Category Share
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------

df10 = pd.read_csv(TOP10_CSV, index_col=0)
df10 = df10.sort_values('listings', ascending=False).reset_index(drop=True).head(10)
df_top10 = df10.groupby('country').sum()

colors = ['#D62828', '#003049', '#F77F00']

palette = {
  'japanese':colors[1],
  'malaysian':colors[2],
  'german':colors[0]
}

bar_chart = go.Figure(data=[
    go.Bar(
        y=df10[df10['country'] == country]['brand'],
        x=df10[df10['country'] == country]['listings'],
        customdata=df10[df10['country'] == country]['country'],
        text=df10[df10['country'] == country]['listings'],
        name=country,
        orientation='h',
        marker=dict(color=palette[country])
    )
    for country in df10['country'].unique()
])

bar_chart.update_layout(
    title='Top 10 Brands by Listing Volume',
    font=dict(size=10, family='system-ui, -apple-system, sans-serif', color='#333333'),
    barmode='overlay',
    plot_bgcolor='white', 
    xaxis_title='listings',
    yaxis_title='brand',
    xaxis_tickformat=',.0f',
    legend=dict(title='country'),
    yaxis=dict(
        title_standoff=10,
        categoryorder='total ascending'
    )
)

bar_chart.update_traces(
    hovertemplate=(
        "<b>%{y}</b><br>"
        "Listings: %{x:,.0f}<br>"
        "Make Origin: %{customdata}"
        "<extra></extra>"
    )
)


pie_chart = go.Figure(
data=[
    go.Pie(
    labels=df_top10.index,
    values=df_top10['listings'],
    direction='counterclockwise',
    marker=dict(
        colors=colors,
        line=dict(color='white', width=2)
    ),
    hole=0.3
    )
]
)

pie_chart.update_layout(
    title='Listing Share by Make Origin',
    showlegend=True,
    paper_bgcolor='white',
    font=dict(size=10, family='system-ui, -apple-system, sans-serif', color='#333333')
)

pie_chart.update_traces(
    hovertemplate=(
        "<b>%{label}</b><br>"
        "Listings: %{value:,}<br>"
        "Share: %{percent}"
        "<extra></extra>"
    )
)


# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# SECTION 3: Brand-Model Metrics Table
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------

def _mode_or_blank(series: pd.Series):
    s = series.dropna()
    return "" if s.empty else s.mode().iloc[0]

def build_brand_model_metrics_df(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    """
    Returns a matrix DataFrame with rows = metrics and columns = models,
    matching your screenshot (Toyota example).
    """
    d = df[df["brand"] == brand].copy()
    if d.empty:
        return pd.DataFrame()

    cols_present = d.columns

    # Order models by listing count desc to put popular ones first
    models = (
        d.groupby("model")
         .size()
         .sort_values(ascending=False)
         .index.astype(str)
         .tolist()
    )

    rows = ["No. of Listings", "Median Price", "Median Age", "Median Mileage"]
    if "type" in cols_present:    rows.append("Type")
    if "segment" in cols_present: rows.append("Segment")

    data = {m: {} for m in models}
    for m in models:
        dm = d[d["model"] == m]
        n = int(len(dm))

        med_price   = float(dm["price"].median())   if "price"   in cols_present else np.nan
        med_age     = float(dm["age"].median())     if "age"     in cols_present else np.nan
        med_mileage = float(dm["mileage"].median()) if "mileage" in cols_present else np.nan
        type_val    = _mode_or_blank(dm["type"])    if "type"    in cols_present else ""
        seg_val     = _mode_or_blank(dm["segment"]) if "segment" in cols_present else ""

        data[m]["No. of Listings"] = n
        if "price" in cols_present:
            data[m]["Median Price"] = med_price
        if "age" in cols_present:
            data[m]["Median Age"] = med_age
        if "mileage" in cols_present:
            data[m]["Median Mileage"] = med_mileage
        if "type" in cols_present:
            data[m]["Type"] = str(type_val)
        if "segment" in cols_present:
            data[m]["Segment"] = str(seg_val)

    # Build matrix with metrics as index (rows)
    df_out = pd.DataFrame(data).reindex(index=rows)

    return df_out


# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# SECTION 4.1: Model Depreciation Curve
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------

def make_depr_figure(df_sub):
    """Return a Plotly figure for the depreciation scatter + fitted curve."""
    if df_sub is None or df_sub.empty:
        fig = go.Figure()
        fig.update_layout(
            # dragmode='pan',
            template="plotly_white",
            title="Depreciation curve",
            xaxis_title="Age (Years)",
            yaxis_title="Price (MYR)",
        )
        fig.add_annotation(text="Select a Brand and Model.", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False)
        return fig

    car_model = f"{df_sub['brand'].iloc[0]} {df_sub['model'].iloc[0]}".title()
    x_data = df_sub['age'].astype(float)
    y_data = df_sub['price'].astype(float)
    mileage_data = df_sub['mileage'].astype(float) if 'mileage' in df_sub.columns else None

    # Fit
    try:
        a0 = np.nanpercentile(y_data, 90)
        b0 = 0.12
        p_optimal, _ = curve_fit(lambda t, a, b: a*np.exp(-b*t), x_data, y_data,
                                 p0=[a0, b0], bounds=(0, np.inf), maxfev=20000)
        initial_val, growth_rate = p_optimal
        depr_20_percent = -np.log(0.80) / growth_rate
        depr_50_percent = -np.log(0.50) / growth_rate
    except Exception:
        initial_val = growth_rate = depr_20_percent = depr_50_percent = np.nan

    # x for fitted curve
    x_sorted = np.linspace(0, max(float(x_data.max()), (depr_50_percent or 0)*1.25, 1.0), 400)
    y_fitted = initial_val*np.exp(-growth_rate*x_sorted) if np.isfinite(growth_rate) else None

    fig = go.Figure()

    # Scatter plot
    fig.add_trace(go.Scatter(
        x=x_data, y=y_data, customdata=mileage_data, mode='markers',
        marker=dict(
            size=8,
            color=df_sub['mileage'] if 'mileage' in df_sub.columns else None,
            colorscale='RdYlBu_r', showscale=('mileage' in df_sub.columns),
            colorbar=dict(title="Mileage (km)", thickness=15, len=0.7)
        ),
        name='Vehicle Data',
        hovertemplate="Age %{x:.2f} yr<br>MYR %{y:,.0f}<br>Mileage: %{customdata:,.0f} km<extra></extra>"
    ))

    # Fitted curve (if fit succeeded)
    if y_fitted is not None:
        pct_drop_fitted = 100.0 * (1.0 - (y_fitted / float(initial_val)))

        fig.add_trace(go.Scatter(
            x=x_sorted, 
            y=y_fitted, 
            mode='lines',
            line=dict(color='black', width=2), 
            name='Fitted Curve',
            customdata=pct_drop_fitted,
            hovertemplate=(
                "Age: %{x:.2f} yr<br>"
                "Est Price: MYR %{y:,.0f}<br>"
                "Drop: %{customdata:.1f}%<extra></extra>"
            )
        ))

        # Red dashed lines at 80% and 50% of initial price
        x0, x1 = float(np.min(x_sorted)), float(np.max(x_sorted))
        xr = x1 - x0
        x_mid = x0 + xr *0.6
        y_80 = 0.80 * float(initial_val)   # 20% drop threshold
        y_50 = 0.50 * float(initial_val)   # 50% drop threshold

        # 80% line with inline text
        fig.add_trace(go.Scatter(
            x=[x0, x_mid, x1],
            y=[y_80, y_80, y_80],
            mode="lines+text",
            textposition="top right",
            line=dict(color="salmon", width=1.5, dash="dash"),
            text=["", f"20% drop · MYR {y_80:,.0f}", ""],
            textfont=dict(color="salmon", size=9),
            hovertemplate="Price threshold: MYR %{y:,.0f}<extra></extra>",
            showlegend=False
        ))

        # 50% line with inline text
        fig.add_trace(go.Scatter(
            x=[x0, x_mid, x1],
            y=[y_50, y_50, y_50],
            mode="lines+text",
            textposition="top right",
            line=dict(color="salmon", width=1.5, dash="dash"),
            text=["", f"50% drop · MYR {y_50:,.0f}", ""],
            textfont=dict(color="salmon", size=9),
            hovertemplate="Price threshold: MYR %{y:,.0f}<extra></extra>",
            showlegend=False
        ))

        ann_text = (
            f"<b>Est. Initial Value:</b> MYR {initial_val:,.0f}<br>"
            f"<b>Time to −20%:</b> {depr_20_percent:.2f} yr<br>"
            f"<b>Time to −50%:</b> {depr_50_percent:.2f} yr"
        )
        fig.add_annotation(
            text=ann_text, xref='paper', yref='paper', x=0.475, y=1,
            showarrow=False, bgcolor='white', bordercolor='black', borderwidth=1,
            font=dict(size=10), align='left'
        )
    else:
        fig.add_annotation(
            text="Not enough data to fit curve.", x=0.5, y=0.5,
            xref='paper', yref='paper', showarrow=False
        )

    fig.update_layout(
        font=dict(size=10),
        title=f"Depreciation Curve — <b>{car_model}</b>",
        xaxis_title='Age (Years)', 
        yaxis_title='Price (MYR)',
        template='plotly_white', 
        hovermode='closest',
        uirevision="stay", 
        legend=dict(x=0.925, y=1),
        margin=dict(l=50, r=20, t=50, b=90),
        # dragmode='pan',
    )

    fig.add_annotation(
        x=0.0, y=-0.25, xref="paper", yref="paper",
        text="Exponential fit on listing prices; time to −20% and −50%.",
        showarrow=False,
        align="left",
        font=dict(size=8, color="#666")
    )
    return fig


# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# SECTION 4.2: Average Contour Plot
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------


# ---------- helpers ----------
def _ensure_age(df, current_year=2025):
    if 'age' in df.columns:
        d = df.copy()
    elif 'year' in df.columns:
        d = df.copy()
        d['age'] = current_year - d['year']
    else:
        raise ValueError("Need 'age' or 'year' column.")
    return d[(d['age'] >= 0) & (d['age'] <= 25)]

def _clean(df, price_min=5_000, price_max=500_000, mileage_min=0, mileage_max=400_000):
    d = df.dropna(subset=['price', 'age', 'mileage'])
    d = d[(d['price'] >= price_min) & (d['price'] <= price_max)]
    d = d[(d['mileage'] >= mileage_min) & (d['mileage'] <= mileage_max)]
    return d

def _bin_edges_age():     return [0,1,2,3,4,5,6,8,10,12,15,20, np.inf]          # yrs
def _bin_edges_mileage(): return list(range(0, 220_000, 20_000)) + [np.inf]     # km

# ---------- main (MEAN-based) ----------
def make_price_contour_plotly_mean(
    df, brand=None, model=None,
    min_count=10,                # minimum listings per bin before smoothing
    smooth_age_bw=1.0,           # Gaussian bandwidth in years
    smooth_mileage_bw=15_000,    # Gaussian bandwidth in km
    clip_quantiles=(0.05, 0.95),
    title_prefix="Average Price Contours"
):
    """
    Plotly contour of AVERAGE price over (Age × Mileage).
    - Filters by brand and optional model
    - Bins to MEAN price per cell, masks sparse cells, optional Gaussian smoothing
    """
    # Empty state
    if not brand:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            title=f"{title_prefix} — select a brand",
            xaxis_title="Mileage (km)", yaxis_title="Age (years)"
        )
        return fig

    d = _clean(_ensure_age(df))
    d = d[d['brand'] == brand]
    if model:
        d = d[d['model'] == model]

    if d.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            title=f"{title_prefix} — {brand}" + (f" {model}" if model else ""),
            xaxis_title="Mileage (km)", yaxis_title="Age (years)"
        )
        fig.add_annotation(text="No data for this selection.", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False)
        return fig

    # --- 1) Bin to grid: MEAN price & counts ---
    age_edges = _bin_edges_age()
    mil_edges = _bin_edges_mileage()

    d['age_bin'] = pd.cut(d['age'], bins=age_edges, include_lowest=True)
    d['mil_bin'] = pd.cut(d['mileage'], bins=mil_edges, include_lowest=True)

    pivot_mean = d.pivot_table(index='age_bin', columns='mil_bin', values='price', aggfunc='mean')
    counts     = d.pivot_table(index='age_bin', columns='mil_bin', values='price', aggfunc='size')

    pivot_mean = pivot_mean.sort_index().sort_index(axis=1)
    counts     = counts.reindex_like(pivot_mean)

    Z = pivot_mean.values.astype(float)
    C = counts.values.astype(float)

    # --- 2) Mask sparse cells ---
    Z_masked = Z.copy()
    Z_masked[(C < min_count) | ~np.isfinite(Z_masked)] = np.nan

    # --- 3) Gaussian smoothing (optional but recommended for mean surface) ---
    # convert real-unit bandwidths to grid sigmas
    age_widths = np.diff([e if np.isfinite(e) else 25 for e in age_edges])
    age_step = float(np.median(age_widths[age_widths > 0]))
    mil_widths = np.diff([e if np.isfinite(e) else 220_000 for e in mil_edges])
    mil_step = float(np.median(mil_widths[mil_widths > 0]))

    sigma_age = smooth_age_bw / age_step
    sigma_mil = smooth_mileage_bw / mil_step

    valid = np.isfinite(Z_masked).astype(float)
    Z_filled = np.where(np.isfinite(Z_masked), Z_masked, 0.0)

    num = gaussian_filter(Z_filled, sigma=(sigma_age, sigma_mil), mode="nearest")
    den = gaussian_filter(valid,    sigma=(sigma_age, sigma_mil), mode="nearest")
    Z_smooth = np.where(den > 1e-6, num / den, np.nan)

    # --- 4) Color clipping for readability ---
    finite_vals = Z_smooth[np.isfinite(Z_smooth)]
    if finite_vals.size == 0:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            title=f"{title_prefix} — {brand}" + (f" {model}" if model else ""),
            xaxis_title="Mileage (km)", yaxis_title="Age (years)"
        )
        fig.add_annotation(text="Insufficient coverage to draw contours.",
                           x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig

    vmin = float(np.quantile(finite_vals, clip_quantiles[0]))
    vmax = float(np.quantile(finite_vals, clip_quantiles[1]))

    # centers for axes
    age_centers = []
    for i in range(len(age_edges)-1):
        a, b = age_edges[i], age_edges[i+1]
        if np.isinf(b): b = 25
        age_centers.append((a + b) / 2.0)
    mil_centers = []
    for i in range(len(mil_edges)-1):
        a, b = mil_edges[i], mil_edges[i+1]
        if np.isinf(b): b = 220_000
        mil_centers.append((a + b) / 2.0)

    # Contour plot
    fig = go.Figure(data=go.Contour(
        z=Z_smooth,
        x=mil_centers,
        y=age_centers,
        contours=dict(coloring='heatmap', showlabels=True, labelfont=dict(size=10)),
        colorscale='RdYlBu_r',
        colorbar=dict(title="Average price (MYR)"),
        zmin=vmin, zmax=vmax,
        connectgaps=False
    ))

    fig.update_layout(
        font=dict(size=10),
        template="plotly_white",
        title=f"{title_prefix} — <b>{brand.title()}</b>" + (f"<b> {model.title()}</b>" if model else ""),
        xaxis_title="Mileage (km)",
        yaxis_title="Age (years)",
        margin=dict(l=50, r=20, t=50, b=90),
        hoverlabel=dict(bgcolor="rgba(0,0,0,0.85)", font_color="white", font_size=12),
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{z:,.0f} MYR</b><br>"
            "Mileage: %{x:,.0f} km<br>"
            "Age: %{y:.1f} years"
            "<extra></extra>"
        )
    )

    fig.add_annotation(
        x=0.0, y=-0.25, xref="paper", yref="paper",
        text="Iso-price lines across age × mileage. Based on binned average prices with smoothing.",
        showarrow=False,
        align="left",
        font=dict(size=8, color="#666")
    )
    return fig

# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# SECTION 5: Compare Depreciation Rate Between Vehicle Models
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------

def make_percent_drop_compare_lines(df, brand1=None, model1=None, brand2=None, model2=None):
    """
    Lines-only % drop vs age for up to two models.
    Uses curve_fit on price = a * exp(-b * age) (same as your depreciation fit).
    Annotations are rendered inside fixed-size boxes above the plot so they align.
    """
    fig = go.Figure()
    ann_texts = []   # list of (text, color)

    def fit_exp_and_curve(df_sub):
        if df_sub is None or df_sub.empty:
            return None, None, None, None
        x_data = df_sub['age'].astype(float).values
        y_data = df_sub['price'].astype(float).values
        try:
            a0, b0 = np.nanpercentile(y_data, 90), 0.12
            p_opt, _ = curve_fit(lambda t, a, b: a*np.exp(-b*t),
                                 x_data, y_data, p0=[a0, b0],
                                 bounds=(0, np.inf), maxfev=20000)
            a, b = float(p_opt[0]), float(p_opt[1])
            if not (np.isfinite(b) and b > 0):
                return None, None, None, None
            d20 = -np.log(0.80) / b
            d50 = -np.log(0.50) / b
            return b, a, float(d20), float(d50)
        except Exception:
            return None, None, None, None

    def add_model_curve(brand, model, color):
        if not (brand and model):
            return False
        df_sub = df[(df['brand'] == brand) & (df['model'] == model)]
        b, a, d20, d50 = fit_exp_and_curve(df_sub)
        if b is None:
            return False

        # x-range for smooth curve
        x_max = max(5.0, float(np.nanmax(df_sub['age']))) if df_sub['age'].size else 5.0
        if np.isfinite(d50):
            x_max = max(x_max, 1.25 * d50)
        x_sorted = np.linspace(0, x_max, 400)
        y_drop = 100.0 * (1.0 - np.exp(-b * x_sorted))

        fig.add_trace(go.Scatter(
            x=x_sorted, y=y_drop, mode='lines',
            line=dict(color=color, width=3),
            name=f"{brand} {model}".title(),
            hovertemplate="Age %{x:.2f} yr<br>Drop %{y:.1f}%<extra></extra>"
        ))

        # collect text for fixed-size box
        # Build annotation text
        parts = [f"<b>Est. Initial Value:</b> MYR {a:,.0f}"] if np.isfinite(a) else []
        if np.isfinite(d20):
            parts.append(f"<b>−20%:</b> {d20:.2f} yr")
        if np.isfinite(d50):
            parts.append(f"<b>−50%:</b> {d50:.2f} yr")
        if parts:
            ann_texts.append((" | ".join(parts), color))
        return True

    plotted_any = False
    if brand1 and model1:
        plotted_any = add_model_curve(brand1, model1, 'slategray') or plotted_any
    if brand2 and model2:
        plotted_any = add_model_curve(brand2, model2, 'salmon') or plotted_any

    if ann_texts:
        y_positions = [1.09, 1.05]
        for idx, (text, color) in enumerate(ann_texts[:2]):
            fig.add_annotation(
                x=0.5, xref='paper', xanchor='center',
                y=y_positions[idx], yref='paper', yanchor='top',
                text=text, showarrow=False, align='center',
                font=dict(size=10, color=color)  # second model will be red
            )

    fig.update_layout(
        title=("Price Drop vs Age — Comparison" if plotted_any else "Price Drop vs Age — Select model(s)"),
        font=dict(size=10),
        template="plotly_white",
        xaxis_title="Age (Years)",
        yaxis_title="Drop from Original Price (%)",
        yaxis=dict(ticksuffix="%"),
        hovermode="closest",
        uirevision="stay",
        height=470,
        showlegend=plotted_any,
        margin=dict(l=60, r=30, t=75, b=60),
    )
    return fig


# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# SECTION 6: State-Level Market Metrics
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------

def compute_state_metrics(df_sub):
    d = df_sub.copy()
    d = d.dropna(subset=["state", "price", "age"])
    d["price"] = pd.to_numeric(d["price"], errors="coerce")
    d["age"]   = pd.to_numeric(d["age"], errors="coerce")
    if "mileage" in d.columns:
        d["mileage"] = pd.to_numeric(d["mileage"], errors="coerce")

    g = d.groupby("state", as_index=False)

    metrics = g.agg(
        n_listings   = ("price", "size"),
        median_price = ("price", "median"),
        mean_price   = ("price", "mean"),
        median_age   = ("age",   "median"),
        mean_age     = ("age",   "mean"),
        median_mileage = ("mileage", "median") if "mileage" in d.columns else ("price", "size")
    )

    # If mileage wasn't present, drop the placeholder column
    if "mileage" not in d.columns and "median_mileage" in metrics.columns:
        metrics = metrics.drop(columns=["median_mileage"])

    # numeric cleanup/rounding
    for c in ["median_price","mean_price","median_age","mean_age","median_mileage"]:
        if c in metrics.columns:
            metrics[c] = pd.to_numeric(metrics[c], errors="coerce")

    return metrics

STATE_METRICS = compute_state_metrics(df)

# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------

metric_options = [
    {"label": "Median Price (MYR)", "value": "median_price"},
    {"label": "Mean Price (MYR)",   "value": "mean_price"},
    {"label": "Median Age (years)", "value": "median_age"},
    {"label": "Mean Age (years)",   "value": "mean_age"},
    {"label": "Listings (count)",   "value": "n_listings"},
    {"label": "Median Mileage (km)", "value": "median_mileage"}
]

METRIC_LABEL = {o["value"]: o["label"] for o in metric_options}

brands = ["All"] + sorted(df["brand"].dropna().unique().tolist())
# dependent models will be built dynamically

dropdown_style = {
    "maxWidth": 420,
    "border": "2px solid transparent",
    "borderRadius": "16px",
    "backgroundColor": "antiquewhite",
    "boxShadow": (
        "0 12px 24px rgba(0, 0, 0, 0.12),"
        "0 4px 8px rgba(0, 0, 0, 0.08)"
    ),
    "fontFamily": "system-ui, -apple-system, sans-serif",
    "fontSize": "14px",
    "padding": "4px",
    # "backgroundColor": "#384457",      # lighter navy-gray
    # "boxShadow": (
    #     "0 14px 28px rgba(0, 0, 0, 0.25),"
    #     "0 6px 12px rgba(0, 0, 0, 0.15)"
    # ),
    # "backgroundColor": "#2b3444",   # dark navy/gray
    # "boxShadow": (
    #     "0 14px 28px rgba(0, 0, 0, 0.2),"   # main outer shadow (downward)
    #     "0 6px 10px rgba(0, 0, 0, 0.2),"    # secondary soft shadow
    #     "0 0 0 1px rgba(255, 255, 255, 0.02)"  # faint outline for definition
    # ),
}

section_icon_colors = {
    "intro": "#2563eb",
    "market": "#f97316",
    "brand": "#0f766e",
    "depreciation": "#7c3aed",
    "compare": "#dc2626",
    "state": "#0ea5e9",
}


# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# DASH APP & UI
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------

fontawesome_stylesheet = "https://use.fontawesome.com/releases/v5.8.1/css/all.css"
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX, fontawesome_stylesheet], suppress_callback_exceptions=True)
app.title = "MY Used-Car Market"
server = app.server

def build_layout():
    return dbc.Container(
        [
            # ------------------------------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------------------------------------------------
            # SECTION 1: Project Introduction & Overview
            # ------------------------------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------------------------------------------------
            html.Div([
                
                html.Div([
                    html.Div('USED CAR PRICING & DEPRECIATION'),
                    html.Div('MALAYSIA OVERVIEW'),
                ],
                style={
                        'textAlign':'center',
                        'fontSize':'48px',
                        'fontWeight':300,
                        'marginTop':'80px',
                        'marginBottom':'80px',
                    },
                ),
                html.Hr(style={'borderColor': '#9ca3af', 'borderWidth': '2px', "marginTop":"35px", "marginBottom":"20px"}),
                html.Div(
                    [
                        html.I(className="fas fa-info-circle", style={"marginRight": "6px", "color": section_icon_colors["intro"]}),
                        html.H3('Project Introduction & Overview:', style={'fontWeight': 'bold', 'margin':0}),
                    ],
                    style={
                        "display": "flex", 
                        "alignItems": "center", 
                        "gap": "6px",
                        "marginBottom":"20px",
                    },
                ),
                
                html.P('''
                This dashboard presents a data-driven analysis of used car depreciation trends in Malaysia, 
                based on listings scraped from Mudah.my during 2022.
                ''', style={'fontSize': '85%'}),  

                html.P('''
                The objective of this project is to uncover how vehicle price, 
                age, and mileage interact across brands and models, 
                providing insights into market behavior, value retention, and resale trends from a quantitative perspective.
                ''', style={'fontSize': '85%'}),
            ]),
            html.Hr(style={'borderColor': '#9ca3af', 'borderWidth': '2px', "marginTop":"35px", "marginBottom":"20px"}),

            # ------------------------------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------------------------------------------------
            # SECTION 2: Market Coverage
            # ------------------------------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------------------------------------------------

            html.Div(
                    [
                        html.I(className="fas fa-chart-pie", style={"marginRight": "6px", "color": section_icon_colors['market']}),
                        html.H3('Market Coverage — Top Brands & Category Share:', style={'fontWeight': 'bold', 'margin':0}),
                    ],
                    style={
                        "display": "flex", 
                        "alignItems": "center", 
                        "gap": "6px",
                        "marginBottom":"20px",
                    },
                ),
            html.P('''
                This study aggregates Mudah.my listings for the top 10 brands and 
                representative models across major vehicle segments (A/B/C/D, SUV, trucks). 
                Car models that have <100 listings and were launched <5 years ago were excluded 
                to ensure that estimates of depreciation are statistically stable. 
                ''', style={'fontSize': '85%'}),

            html.P('''
                The final dataset comprises 50k+ records covering 50+ popular car models, 
                with price, year/age, mileage, location, and segment.
                ''', style={'fontSize': '85%'}),

            html.P('''
                This section provides a macro-level snapshot of the used-car market composition.    
                ''', style={'fontSize': '85%'}),

            dbc.Row([
                dbc.Col(dcc.Graph(figure=bar_chart, responsive=True,
                                config={"displayModeBar": "hover", "displaylogo": False},
                                style={"height": "450px"}), xs=12, sm=12, md=6, lg=6, xl=6),
                dbc.Col(dcc.Graph(figure=pie_chart, responsive=True,
                                config={"displayModeBar": "hover", "displaylogo": False},
                                style={"height": "450px"}), xs=12, sm=12, md=6, lg=6, xl=6),
            ],),

            html.Div([
                html.P('''
                    Comparative distribution of vehicle listings by brand and make origin, 
                    highlighting dominant market contributors and regional manufacturing influence.                   
                    ''', style={'fontSize': '75%'}),
            ], style={'textAlign': 'center'}),

            html.Br(),

            html.Div([
                html.P('*Figures reflect listing counts on Mudah.my during the collection window.', style={'fontSize': '65%'}),
            ], style={'textAlign': 'right'}),
            
            html.Hr(style={'borderColor': '#9ca3af', 'borderWidth': '2px', "marginTop":"35px", "marginBottom":"20px"}),

            # ------------------------------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------------------------------------------------
            # SECTION 3: Brand Coverage & Model Stats
            # ------------------------------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------------------------------------------------

            html.Div(
                    [
                        html.I(className="fas fa-car", style={"marginRight": "6px", "color": section_icon_colors['brand']}),
                        html.H3('Brand Coverage and Model Stats:', style={'fontWeight': 'bold', 'margin':0}),
                    ],
                    style={
                        "display": "flex", 
                        "alignItems": "center", 
                        "gap": "6px",
                        "marginBottom":"20px",
                    },
                ),
            html.P('Popular car model listings for each brand were scraped and cleaned by fitting a robust model of price vs age & mileage for each brand-model, points with unusually large residuals (MAD-based) were removed, keeping only typical market prices.', style={'fontSize': '85%'}),
            html.P('''
                Explore performance metrics and respective depreciation behaviour across all available models.
                ''', style={'fontSize': '85%'}),
            html.Br(),

            dbc.Row([
                dbc.Col([
                    html.P("Choose a brand:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id="brand-general",
                        options=[{"label": b.title(), "value": b} for b in sorted(df["brand"].dropna().unique())],
                        value=sorted(df["brand"].dropna().unique())[0],
                        style=dropdown_style,
                        clearable=False,
                        className="bold-dd",
                    ),
                ], xs=6, sm=6, md=2, lg=2, xl=2),
                dbc.Col([
                    html.Div(id="brand-models-table"),
                    html.Br(),
                    html.P('Model metrics from Mudah.my listings: n, median price, age, mileage.', style={'fontSize': '60%'}),
                ], xs=12, sm=12, md=10, lg=10, xl=10),
            ]),

            html.Hr(style={'borderColor': '#9ca3af', 'borderWidth': '2px', "marginTop":"35px", "marginBottom":"20px"}),

            # ------------------------------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------------------------------------------------
            # SECTION 4: Depreciation Curve & Contour Plot
            # ------------------------------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------------------------------------------------

            html.Div(
                    [
                        html.I(className="fas fa-chart-line", style={"marginRight": "6px", "color": section_icon_colors['depreciation']}),
                        html.H3('Model Value & Depreciation Insights:', style={'fontWeight': 'bold', 'margin':0}),
                    ],
                    style={
                        "display": "flex", 
                        "alignItems": "center", 
                        "gap": "6px",
                        "marginBottom":"20px",
                    },
                ),
            html.P('Explore pricing behavior by age and mileage for the selected model.', style={'fontSize': '85%'}),
            html.Br(),

            dbc.Row([
                dbc.Col([
                    html.P('Choose a model:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id="model-general",
                        style=dropdown_style,
                        clearable=False,
                        className="bold-dd",
                    ),
                ], xs=6, sm=6, md=2, lg=2, xl=2),
                dbc.Col([
                    dcc.Graph(
                        id="depr-graph", 
                        responsive=True,
                        config={"displayModeBar": "hover", "displaylogo": False},
                        style={"height": "600px"}
                    ),
                ], xs=12, sm=12, md=5, lg=5, xl=5),
                dbc.Col([
                    dcc.Graph(id="age-mileage-contour", config={"displayModeBar": "hover", "displaylogo": False}),
                ],  xs=12, sm=12, md=5, lg=5, xl=5)
            ]),

            html.Hr(style={'borderColor': '#9ca3af', 'borderWidth': '2px', "marginTop":"35px", "marginBottom":"20px"}),

            # ------------------------------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------------------------------------------------
            # SECTION 5: Compare Depreciation Between Models
            # ------------------------------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------------------------------------------------

            html.Div(
                    [
                        html.I(className="fas fa-balance-scale", style={"marginRight": "6px", "color": section_icon_colors['compare']}),
                        html.H3('Compare Depreciation Rate Between Vehicle Models:', style={'fontWeight': 'bold', 'margin':0}),
                    ],
                    style={
                        "display": "flex", 
                        "alignItems": "center", 
                        "gap": "6px",
                        "marginBottom":"20px",
                    },
                ),
            html.Br(),

            dbc.Row([
                dbc.Col([], xs=0, sm=0, md=3, lg=3, xl=3),
                dbc.Col([
                    html.P('Model 1:', style={'fontWeight': 'bold'}),
                    html.Div([
                        dcc.Dropdown(
                            id="brand-1",
                            options=[{"label": b.title(), "value": b} for b in sorted(df["brand"].dropna().unique())],
                            value='honda',
                            placeholder="Select brand",
                            style=dropdown_style,
                            className="bold-dd",
                        ),
                    ], style={'width': '80%'}),
                    html.Div([
                        dcc.Dropdown(
                            id="model-1", 
                            options=[], 
                            value='city',
                            placeholder="Select model",
                            style=dropdown_style,
                            className="bold-dd",
                        ),
                    ], style={'width': '80%'}),
                ], xs=6, sm=6, md=3, lg=3, xl=3, style={
                                "display": "flex",
                                "flexDirection": "column",   # stack P on top of dropdown
                                "alignItems": "center",      # ⬅ center children horizontally
                            }
                ),
                dbc.Col([
                    html.P('Model 2:', style={'fontWeight': 'bold'}),
                    html.Div([
                        dcc.Dropdown(
                            id="brand-2",
                            options=[{"label": b.title(), "value": b} for b in sorted(df["brand"].dropna().unique())],
                            value='toyota',
                            placeholder="Select brand",
                            style=dropdown_style,
                            className="bold-dd",
                        ),
                    ], style={'width': '80%'}),
                    html.Div([
                        dcc.Dropdown(
                            id="model-2", 
                            options=[], 
                            value='vios',
                            placeholder="Select model",
                            style=dropdown_style,
                            className="bold-dd",
                        ),
                    ], style={'width': '80%'}),
                ], xs=6, sm=6, md=3, lg=3, xl=3, style={
                                "display": "flex",
                                "flexDirection": "column",   # stack P on top of dropdown
                                "alignItems": "center",      # ⬅ center children horizontally
                            }
                ),
                dbc.Col([], xs=0, sm=0, md=3, lg=3, xl=3),
            ]),

            html.Br(),

            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="drop-compare-fig", config={"displayModeBar": "hover", "displaylogo": False}),
                ], xs=12, sm=12, md=6, lg=6, xl=6),
            ], justify='center'),

            html.Hr(style={'borderColor': '#9ca3af', 'borderWidth': '2px', "marginTop":"35px", "marginBottom":"20px"}),

            # ------------------------------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------------------------------------------------
            # SECTION 6: State-Level Metrics
            # ------------------------------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------------------------------------------------

            html.Div(
                    [
                        html.I(className="fas fa-map-marked-alt", style={"marginRight": "6px", "color": section_icon_colors['state']}),
                        html.H3('State-Level Market Metrics:', style={'fontWeight': 'bold', 'margin':0}),
                    ],
                    style={
                        "display": "flex", 
                        "alignItems": "center", 
                        "gap": "6px",
                        "marginBottom":"20px",
                    },
                ),
            html.P('''
                Compare listings, prices, age, and mileage across Malaysian states. 
                ''', style={'fontSize': '85%'}),
            html.P('''
                State-level metrics show where supply is concentrated and how typical prices, vehicle age, and mileage differ by location. 
                The bar ranks states by the selected metric whilst the map presents the same data spatially.
                ''', style={'fontSize': '85%'}),
            html.Br(),
            dbc.Row([
                dbc.Col([], xs=0, sm=0, md=3, lg=3, xl=3),

                dbc.Col([
                    html.P("Select Brand:", style={'fontWeight': 'bold'}),
                    html.Div([
                        dcc.Dropdown(options=[{"label": b.title(), "value": b} for b in brands],
                                    value="All",
                                    id="brand-dd",
                                    clearable=False,
                                    style=dropdown_style,
                                    className="bold-dd",
                        ),
                    ], style={'width': '100%'}),
                ], xs=6, sm=6, md=3, lg=3, xl=3, style={
                                "display": "flex",
                                "flexDirection": "column",   # stack P on top of dropdown
                                "alignItems": "center",      # ⬅ center children horizontally
                            }
                ),

                dbc.Col([
                    html.P("Select Model:", style={'fontWeight': 'bold'}),
                    html.Div([
                        dcc.Dropdown(options=[{"label": "All", "value": "All"}],
                                    value="All",
                                    id="model-dd",
                                    clearable=False,
                                    style=dropdown_style,
                                    className="bold-dd",
                        ),
                    ], style={'width': '100%'}),
                ], xs=6, sm=6, md=3, lg=3, xl=3, style={
                                "display": "flex",
                                "flexDirection": "column",   # stack P on top of dropdown
                                "alignItems": "center",      # ⬅ center children horizontally
                            }
                ),

                dbc.Col([], xs=0, sm=0, md=3, lg=3, xl=3),
                
            ], className="mt-2 mb-2"),
            
            html.Br(),

            dbc.Row(id='state-brand-model-metrics-cards',
            ),

            dbc.Switch(
                id="agg-mean-toggle",
                label="Toggle to swtich between Mean and Median aggregates",
                value=False,   # False => median (default), True => mean
                className="mb-3",
                style={"fontSize": "75%"}
            ),

            html.Br(),

            dbc.Row([
                dbc.Col([]),
                dbc.Col([
                    html.P("Select Metric:", style={'fontWeight': 'bold'}),
                    html.Div([
                        dcc.Dropdown(metric_options,
                                    value="median_price",
                                    id="metric-dd",
                                    clearable=False,
                                    style=dropdown_style,
                                    className="bold-dd",
                        ),
                    ], style={'width': '100%'}),
                ], xs=6, sm=6, md=3, lg=3, xl=3, style={
                                "display": "flex",
                                "flexDirection": "column",   # stack P on top of dropdown
                                "alignItems": "center",      # ⬅ center children horizontally
                            } 
                ),
                dbc.Col([]),
                            
            ]),

            html.Br(),
           

            dbc.Row([
                dbc.Col(dcc.Graph(id="bar",
                                config={"displayModeBar": "hover", "displaylogo": False},
                                style={"height": "450px"}), xs=12, sm=12, md=5, lg=5, xl=5,),
                dbc.Col(dcc.Graph(id="map", responsive=True,
                                config={"displayModeBar": "hover", "displaylogo": False, "scrollZoom": True},
                                style={"height": "450px"}), xs=12, sm=12, md=7, lg=7, xl=7,),
            ], className="mt-2 mb-2"),

            html.Hr(style={'borderColor': '#9ca3af', 'borderWidth': '2px', "marginTop":"35px", "marginBottom":"20px"}),
            html.Div(
                [
                    'State Metrics Summary for ',
                    html.B(id="state-table-title", style={'fontWeight':700}),
                ],
                style={
                    'fontSize':'18px',
                    'fontWeight':300,
                }
            ),
            dbc.Row([
                dbc.Col(dash_table.DataTable(
                    style_header={"fontWeight": "700", "fontSize":13, "backgroundColor": "#4a5a72", "color":"white", "border": "1px solid #e7e7e7"},
                    style_cell={
                        "border": "1px solid #e7e7e7",
                        "padding": "6px",
                        "fontSize": 12,
                        "whiteSpace": "nowrap",
                        "textOverflow": "ellipsis",
                        "overflow": "hidden",
                        "textAlign": "center",
                        "minWidth": 80, "width": 110, "maxWidth": 220
                    },
                    style_data_conditional=[
                        {"if": {"column_id": "state"}, "fontWeight": "600", "fontSize":13, "backgroundColor": "#d1d5db", "textAlign": "left"},
                    ],
                    id="state-table",
                    style_table={"overflowY": "auto"},
                    sort_action="native",
                )),
            ], className="mt-2 mb-2"),

            html.Br(),

        ],

        fluid=True,
        className="p-5",
    )

app.layout = build_layout()


# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
# CALLBACKS
# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------
# SECTION 3: Brand Coverage Model Stats - Dependant Brand-Model dropdown callback
# --------------------------------------------------------------------------------------------------------------------
@app.callback(
        Output("model-general", "options"),
        Output("model-general", "value"),
        Input("brand-general", "value")
)
def update_models_general(brand):
    if not brand:
        return [], None
    models = sorted(df.loc[df["brand"] == brand, "model"].dropna().unique())
    opts = [{"label": m.title(), "value": m} for m in models]
    val = models[0] if models else None
    return opts, val

# --------------------------------------------------------------------------------------------------------------------
# SECTION 3: Brand Coverage Model Stats - Model metrics table generation callback
# --------------------------------------------------------------------------------------------------------------------
@app.callback(
    Output("brand-models-table", "children"),
    Input("brand-general", "value")
)
def update_brand_models_table(brand):
    if not brand:
        return html.Div("Select a brand to see model metrics.", style={"color": "#666", "padding": "8px"})

    df_matrix = build_brand_model_metrics_df(df, brand)
    if df_matrix.empty:
        return html.Div("No models found for this brand.", style={"color": "#666", "padding": "8px"})

    # Convert to display table and format as strings
    table_df = df_matrix.reset_index().rename(columns={"index": "Metrics"})
    display_df = table_df.copy()

    if "No. of Listings" in display_df["Metrics"].values:
        mask = display_df["Metrics"] == "No. of Listings"
        display_df.loc[mask, display_df.columns != "Metrics"] = (
            display_df.loc[mask, display_df.columns != "Metrics"]
            .applymap(lambda x: f"{int(x):,}" if pd.notna(x) else "")
        )

    if "Median Price" in display_df["Metrics"].values:
        mask = display_df["Metrics"] == "Median Price"
        display_df.loc[mask, display_df.columns != "Metrics"] = (
            display_df.loc[mask, display_df.columns != "Metrics"]
            .applymap(lambda x: f"MYR {x:,.0f}" if pd.notna(x) else "")
        )

    if "Median Age" in display_df["Metrics"].values:
        mask = display_df["Metrics"] == "Median Age"
        display_df.loc[mask, display_df.columns != "Metrics"] = (
            display_df.loc[mask, display_df.columns != "Metrics"]
            .applymap(lambda x: f"{x:.1f} yrs" if pd.notna(x) else "")
        )

    if "Median Mileage" in display_df["Metrics"].values:
        mask = display_df["Metrics"] == "Median Mileage"
        display_df.loc[mask, display_df.columns != "Metrics"] = (
            display_df.loc[mask, display_df.columns != "Metrics"]
            .applymap(lambda x: f"{x:,.0f} km" if pd.notna(x) else "")
        )

    columns = [{"name": "Metrics", "id": "Metrics"}] + [
        {"name": str(c), "id": str(c)} for c in display_df.columns if c != "Metrics"
    ]

    table = DataTable(
        id="brand-models-datatable",
        columns=columns,
        data=display_df.to_dict("records"),
        style_table={"overflowX": "auto"},
        style_header={"fontWeight": "700", "fontSize":13, "backgroundColor": "#4a5a72", "color":"white", "border": "1px solid #e7e7e7"},
        style_cell={
            "border": "1px solid #e7e7e7",
            "padding": "6px",
            "fontSize": 12,
            "whiteSpace": "nowrap",
            "textOverflow": "ellipsis",
            "overflow": "hidden",
            "textAlign": "center",
            "minWidth": 80, "width": 110, "maxWidth": 220
        },
        style_data_conditional=[
            {"if": {"column_id": "Metrics"}, "fontWeight": "600", "fontSize":13, "backgroundColor": "#d1d5db", "textAlign": "left"},
            # {"if": {"filter_query": "{Metrics} = 'No. of Listings'"}, "textAlign": "right"},
            # {"if": {"filter_query": "{Metrics} = 'Median Price'"}, "textAlign": "right"},
            # {"if": {"filter_query": "{Metrics} = 'Median Age'"}, "textAlign": "right"},
            # {"if": {"filter_query": "{Metrics} = 'Median Mileage'"}, "textAlign": "right"},
        ],
        fixed_rows={"headers": True},
        sort_action="native",
        page_action="none",
    )
    return table

# --------------------------------------------------------------------------------------------------------------------
# SECTION 4.1: Depreciation Curve - Graph generation callback
# --------------------------------------------------------------------------------------------------------------------
@app.callback(
    Output("depr-graph", "figure"),
    Input("brand-general", "value"),
    Input("model-general", "value"),
)
def update_depr_graph(brand, model):
    if not brand or not model or brand == "All" or model == "All":
        # show a friendly blank until both are chosen
        return make_depr_figure(df.head(0))
    d = df[(df["brand"] == brand) & (df["model"] == model)]
    return make_depr_figure(d)

# --------------------------------------------------------------------------------------------------------------------
# SECTION 4.1: Contour Plot - Graph generation callback
# --------------------------------------------------------------------------------------------------------------------
@app.callback(
    Output("age-mileage-contour", "figure"),
    Input("brand-general", "value"),
    Input("model-general", "value"),
)
def update_price_contour(brand, model):
    return make_price_contour_plotly_mean(df, brand=brand, model=model)

# --------------------------------------------------------------------------------------------------------------------
# SECTION 5: Compare Depreciation - Brand-Model dependant dropdowns for both comparing models
# --------------------------------------------------------------------------------------------------------------------
@app.callback(
        Output("model-1", "options"),
        Output("model-1", "value"), 
        Input("brand-1", "value")
)
def populate_models_1(brand):
    if not brand:
        return []
    models = sorted(df.loc[df["brand"] == brand, "model"].dropna().unique())
    opts = [{"label": m.title(), "value": m} for m in models]
    val = 'city'
    return opts, val

@app.callback(
        Output("model-2", "options"),
        Output("model-2", "value"), 
        Input("brand-2", "value")
)
def populate_models_2(brand):
    if not brand:
        return []
    models = sorted(df.loc[df["brand"] == brand, "model"].dropna().unique())
    opts = [{"label": m.title(), "value": m} for m in models]
    val = 'vios'
    return opts, val

# --------------------------------------------------------------------------------------------------------------------
# SECTION 5: Compare Depreciation - Graph generation callback
# --------------------------------------------------------------------------------------------------------------------
@app.callback(
    Output("drop-compare-fig", "figure"),
    Input("brand-1", "value"),
    Input("model-1", "value"),
    Input("brand-2", "value"),
    Input("model-2", "value"),
)
def update_compare_fig(b1, m1, b2, m2):
    return make_percent_drop_compare_lines(df, b1, m1, b2, m2)

# --------------------------------------------------------------------------------------------------------------------
# SECTION 6: State Metrics - Dependant Brand-Model dropdown callback
# --------------------------------------------------------------------------------------------------------------------
@app.callback(
    Output("model-dd", "options"),
    Output("model-dd", "value"),
    Input("brand-dd", "value"),
)
def update_models(brand):
    if brand == "All":
        models = ["All"]
    else:
        models = ["All"] + sorted(df.loc[df["brand"] == brand, "model"].dropna().unique().tolist())
    opts = [{"label": m.title(), "value": m} for m in models]
    val = "All" if "All" in models else (models[0] if models else None)
    return opts, val

def _filter_df(brand, model):
    d = df.copy()
    if brand and brand != "All":
        d = d[d["brand"] == brand]
    if model and model != "All":
        d = d[d["model"] == model]
    return d

def _compute_metrics_filtered(brand, model):
    d = _filter_df(brand, model)
    met = compute_state_metrics(d)
    # apply min listings threshold only to T50 validity (keep table rows)
    # met.loc[met["n_listings"] < min_points, "t50"] = np.nan
    return met

# --------------------------------------------------------------------------------------------------------------------
# SECTION 6: State Level Metrics - KPI Cards generation callback
# --------------------------------------------------------------------------------------------------------------------
@app.callback(
    Output('state-brand-model-metrics-cards', 'children'),
    Input("brand-dd", "value"),
    Input("model-dd", "value"),
    Input("agg-mean-toggle", "value"),
)

def update_state_metrics_cards(brand, model, use_mean):
    d = _filter_df(brand, model)
    met = compute_state_metrics(d)

    # total_listings
    total_listings = int(met["n_listings"].sum())

    # choose aggregator
    agg_fn = "mean" if use_mean else "median"
    agg_title = "Mean" if use_mean else "Median"

    # safely aggregate a column if present
    def agg_col(col):
        if col not in met.columns or met[col].dropna().empty:
            return None
        if agg_fn == "mean":
            return float(met[col].mean())
        else:
            return float(met[col].median())

    price_val   = agg_col("median_price")      # table column name
    age_val     = agg_col("median_age")
    mileage_val = agg_col("median_mileage")

    median_price   = int(price_val)   if price_val   is not None else None
    median_age     = float(age_val)   if age_val     is not None else None
    median_mileage = int(mileage_val) if mileage_val is not None else None

    cards = []

    # Define color palette
    colors = {
        'primary': '#3498db',    # Blue
        'success': '#2ecc71',    # Green
        'warning': '#f39c12',    # Orange
        'info': '#9b59b6'        # Purple
    }

    # Base card style with hover effect
    card_style_base = {
        'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
        'borderRadius': '12px',
        'border': 'none',
        'height': '100%',
        'transition': 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
        'overflow': 'hidden'
    }

    header_style = {
        'fontSize': 14,
        'textAlign': 'center',
        'backgroundColor': '#2c3e50',  # Dark navy-gray
        'color': '#ffffff',
        'fontWeight': '600',
        'border': 'none',
        'padding': '14px',
        'letterSpacing': '0.5px'
    }

    body_style = {
        'textAlign': 'center',
        'backgroundColor': '#ffffff'
    }

    value_style = {
        'textAlign': 'center',
        'color': '#2c3e50',
        'fontWeight': '700',
        'margin': '0',
        'fontSize': 20
    }

    card_style_listings = {**card_style_base}
    cards.append(
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardHeader("Total Listings", style=header_style),
                    dbc.CardBody(
                        [
                            html.Div(
                                html.I(className="fas fa-list", style={'fontSize': 24, 'color': colors['primary'], 'marginBottom': '8px'}),
                            ),
                            html.H3(f"{total_listings:,}", style={**value_style, 'color': colors['primary']})
                        ],
                        style=body_style
                    ),
                ],
                style=card_style_listings,
                className='hover-card'
            ),
            width=3,
            className='mb-3'
        )
    )
    # === Price KPI ===
    if median_price is not None:
        cards.append(
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(f"{agg_title} Price", style=header_style),
                        dbc.CardBody(
                            [
                                html.Div(html.I(className="fas fa-dollar-sign",
                                                style={'fontSize': 24, 'color': colors['success'], 'marginBottom': '8px'})),
                                html.H3(f"{median_price:,}", style={**value_style, 'color': colors['success']}),
                                html.Div("MYR", style={'fontSize': 13, 'color': '#95a5a6', 'marginTop': '4px', 'fontWeight': '500'}),
                            ],
                            style=body_style
                        ),
                    ],
                    style=card_style_base,
                    className='hover-card'
                ),
                width=3, className='mb-3'
            )
        )

    # === Age KPI ===
    if median_age is not None:
        cards.append(
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(f"{agg_title} Age", style=header_style),
                        dbc.CardBody(
                            [
                                html.Div(html.I(className="fas fa-calendar-alt",
                                                style={'fontSize': 24, 'color': colors['warning'], 'marginBottom': '8px'})),
                                html.H3(f"{median_age:.1f}", style={**value_style, 'color': colors['warning']}),
                                html.Div("years", style={'fontSize': 13, 'color': '#95a5a6', 'marginTop': '4px', 'fontWeight': '500'}),
                            ],
                            style=body_style
                        ),
                    ],
                    style=card_style_base,
                    className='hover-card'
                ),
                width=3, className='mb-3'
            )
        )

    # === Mileage KPI ===
    if median_mileage is not None:
        cards.append(
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(f"{agg_title} Mileage", style=header_style),
                        dbc.CardBody(
                            [
                                html.Div(html.I(className="fas fa-tachometer-alt",
                                                style={'fontSize': 24, 'color': colors['info'], 'marginBottom': '8px'})),
                                html.H3(f"{median_mileage:,}", style={**value_style, 'color': colors['info']}),
                                html.Div("km", style={'fontSize': 13, 'color': '#95a5a6', 'marginTop': '4px', 'fontWeight': '500'}),
                            ],
                            style=body_style
                        ),
                    ],
                    style=card_style_base,
                    className='hover-card'
                ),
                width=3, className='mb-3'
            )
        )

    return cards

# --------------------------------------------------------------------------------------------------------------------
# SECTION 6: State Level Metrics - Bar/Map Chart, State Metrics Table generation callback
# --------------------------------------------------------------------------------------------------------------------
@app.callback(
    Output("map", "figure"),
    Output("bar", "figure"),
    Output("state-table", "data"),
    Output("state-table", "columns"),
    Output("state-table-title", "children"),
    Input("metric-dd", "value"),
    Input("brand-dd", "value"),
    Input("model-dd", "value"),
)
def update_outputs(metric, brand, model):
    met = _compute_metrics_filtered(brand, model)

    # color range: robust 5–95% to avoid outlier wash-out
    series = met[metric].astype(float)
    if series.notna().sum() >= 3:
        lo, hi = np.nanquantile(series, [0.05, 0.95])
        if lo == hi:
            lo, hi = float(series.min()), float(series.max() + 1e-9)
    else:
        lo, hi = float(series.min()), float(series.max() + 1e-9)

    # Bar (top 10 state by metric)
    met_bar = met.sort_values(metric, ascending=True)

    pretty = {
            "median_price": "Median price",
            "mean_price":   "Mean price",
            "median_age":   "Median age",
            "mean_age":     "Mean age",
            "median_mileage": "Median mileage",
            "mean_mileage":   "Mean mileage",
            "n_listings":     "Total listings",
        }
    fmt = {
        "median_price": (":,.0f", " MYR"),
        "mean_price":   (":,.0f", " MYR"),
        "median_age":   (":.1f",  " yrs"),
        "mean_age":     (":.1f",  " yrs"),
        "median_mileage": (":,.0f", " km"),
        "mean_mileage":   (":,.0f", " km"),
        "n_listings":     (":,d",  ""),
    }

    label  = pretty.get(metric, metric.replace("_", " ").title())
    spec, suffix = fmt.get(metric, (":.2f", ""))   

    state_series = met["state"] if "state" in met.columns else met["state_norm"]

    brand = brand.title()
    model = model.title()

    # Title rules:
    if brand == "All":
        title_text = f"Top States by {label} for <b>All Listings</b>"
        map_title_text = f"{label} for <b>All Listings</b> by State"
        table_title_text = f"All Listings"
    elif model and model != "All":
        title_text = f"Top States by {label} for <b>{brand} {model} Listings</b>"
        map_title_text = f"{label} for <b>{brand} {model} Listings</b> by State"
        table_title_text = f"{brand} {model} Listings"
    else:
        title_text = f"Top States by {label} for <b>{brand} Listings</b>"
        map_title_text = f"{label} for <b>{brand} Listings</b> by State"
        table_title_text = f"{brand} Listings"

    xlab = METRIC_LABEL.get(metric, metric.replace("_", " ").title())

    # Bar
    fig_bar = px.bar(
        met_bar,
        x=metric, y="state", orientation="h",
        title=title_text,
        text=met_bar[metric].round(2),
    )

    fig_bar.update_layout(
        xaxis_title=xlab,
        font=dict(size=10),
        margin=dict(l=50, r=20, t=50, b=50),
        plot_bgcolor='white',
    )

    fig_bar.update_traces(
        marker_color="#4b5563",
        hovertemplate=(
            "<b>%{y}</b><br>"
            f"{label}: %{{x{spec}}}{suffix}"
            "<extra></extra>"
        )
    )

    fig_bar.update_xaxes(tickformat=".0f")

    # Map
    fig_map = px.choropleth_mapbox(
        met.rename(columns={"state": "state_norm"}),
        geojson=MYS,
        locations="state_norm",
        featureidkey=FEATURE_ID_KEY,
        color=metric,
        range_color=(lo, hi) if np.isfinite(lo) and np.isfinite(hi) else None,
        color_continuous_scale=px.colors.sequential.Reds,
        mapbox_style="carto-positron",
        zoom=4.25,
        center={"lat": 4.5, "lon": 109.5},
        opacity=0.80,
        hover_data={
            "state_norm": True,
            "median_price": ":,.0f",
            "median_age": ":.1f",
            "n_listings": ":,",
        },
        title=map_title_text,
    )

    fig_map.update_layout(
        margin=dict(l=50, r=20, t=50, b=50),
        font=dict(size=10),
    )

    fig_map.update_traces(
        customdata=state_series.values, 
        hovertemplate=(
            "<b>%{customdata}</b><br>"
            f"{label}: %{{z{spec}}}{suffix}"
            "<extra></extra>"
        )
    )

    # Table
    table_cols = [
        {"name": "State", "id": "state"},
        {"name": "Listings", "id": "n_listings", "type": "numeric",
         "format": {"locale": {}, "specifier": ","}},
    ]
    if "median_price" in met.columns:
        table_cols.append({"name": "Median Price (MYR)", "id": "median_price",
                           "type": "numeric", "format": {"locale": {}, "specifier": ",.0f"}})
    if "mean_price" in met.columns:
        table_cols.append({"name": "Mean Price (MYR)", "id": "mean_price",
                           "type": "numeric", "format": {"locale": {}, "specifier": ",.0f"}})
    if "median_age" in met.columns:
        table_cols.append({"name": "Median Age (yrs)", "id": "median_age",
                           "type": "numeric", "format": {"specifier": ".1f"}})
    if "mean_age" in met.columns:
        table_cols.append({"name": "Mean Age (yrs)", "id": "mean_age",
                           "type": "numeric", "format": {"specifier": ".1f"}})
    if "median_mileage" in met.columns:
        table_cols.append({"name": "Median Mileage (km)", "id": "median_mileage",
                           "type": "numeric", "format": {"locale": {}, "specifier": ",.0f"}})

    table_data = met.sort_values("state").to_dict("records")

    return fig_map, fig_bar, table_data, table_cols, table_title_text


# ------------------------------------------------------------------------------------------------------------------------
# Run Dash App
# ------------------------------------------------------------------------------------------------------------------------
# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=8050)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8050"))
    app.run(host="0.0.0.0", port=port, debug=False)