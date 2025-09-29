######################################################################################
# Purpose:        Forecast CO₂, CH₄, N₂O, and SF₆ concentrations for the next 6 
#                 months using SARIMA and create an interactive visualization.
#
# Description:    This script loads monthly greenhouse gas data from a public URL,
#                 cleans it, and then iterates through each specified gas. For each 
#                 gas, it performs a grid search to find the optimal SARIMA 
#                 parameters based on the AIC. It then fits the model, generates a 
#                 6-month forecast with a 95% confidence interval, and saves all 
#                 forecasts to a CSV file. Finally, it produces a single interactive 
#                 HTML dot chart using Plotly, allowing for easy comparison and 
#                 analysis of the historical and forecasted data for each gas.
#
# Input File:     - Public Google Sheet CSV accessed via URL.
#
# Output Files:   - CSV: "ghg_sarima_forecasts_next_6_months.csv" (forecast + CI)
#                 - HTML: "index.html" (interactive plot)
#
# Models Used:    - statsmodels.tsa.statespace.SARIMAX
#
# Author:         Alberth Nahas (alberth.nahas@bmkg.go.id)
# Created Date:   2025-07-22
# Version:        2.2.0 (Restored Indonesian language and kept layout enhancements)
######################################################################################

import pandas as pd
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- 1. Load and Prepare Data ---
# URL for the public Google Sheet CSV
url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRjhwQGnOCm51KTrr-xAgLjm1CwIyE9OzSB4WsP8xEvn6YpACXp36ikIMnwqZ2Fyw/pub?gid=1720832544&single=true&output=csv'

# Load the dataset, treating '#N/A' as missing values
try:
    df = pd.read_csv(url, na_values='#N/A')
    print("✅ Data loaded successfully.")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

# Convert 'Date' column to datetime objects using the correct format
df['Date'] = pd.to_datetime(df['Date'], format='%b-%Y')

# Set Date as the index first
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

# Define the target gases we will be working with
target_gases = ['CO2_seasonal', 'CH4_seasonal', 'N2O_seasonal', 'SF6_seasonal']

# Create a subset with only the seasonal columns we need
df_seasonal = df[target_gases].copy()

# Drop rows where ANY of the seasonal columns have missing values
df_seasonal.dropna(inplace=True)

# Use the cleaned seasonal data for the rest of the analysis
df = df_seasonal

# --- 2. SARIMA Forecasting for Each Gas ---
forecast_horizon = 6  # Next 6 months
all_forecasts = []

def optimize_sarima(ts_data, p_range, d_range, q_range, sP_range, sD_range, sQ_range, seasonality):
    """
    Performs a grid search to find the best SARIMA parameters based on AIC.
    """
    best_aic = float('inf')
    best_order = None
    best_seasonal_order = None
    
    param_combinations = list(itertools.product(p_range, d_range, q_range))
    seasonal_param_combinations = list(itertools.product(sP_range, sD_range, sQ_range))
    
    for order in param_combinations:
        for seasonal_order_parts in seasonal_param_combinations:
            seasonal_order = seasonal_order_parts + (seasonality,)
            try:
                model = SARIMAX(ts_data,
                                order=order,
                                seasonal_order=seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                results = model.fit(disp=False)
                
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = order
                    best_seasonal_order = seasonal_order
            except:
                continue
    return best_order, best_seasonal_order

# --- Loop through each gas and perform forecasting ---
print(f"\n--- Memulai Prediksi SARIMA ---")
p = d = q = range(0, 2)
P = D = Q = range(0, 2)
s = 12

for gas in target_gases:
    print(f"\n🔄 Memproses: {gas}")
    ts = df[gas]
    
    best_order, best_seasonal_order = optimize_sarima(ts, p, d, q, P, D, Q, s)
    
    sarima_model = SARIMAX(ts,
                           order=best_order,
                           seasonal_order=best_seasonal_order,
                           enforce_stationarity=False,
                           enforce_invertibility=False)
    sarima_result = sarima_model.fit(disp=False)
    
    forecast = sarima_result.get_forecast(steps=forecast_horizon)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    
    last_data_date = ts.index.max()
    forecast_start_date = last_data_date + pd.DateOffset(months=1)
    forecast_dates = pd.date_range(start=forecast_start_date, periods=forecast_horizon, freq='MS')
    
    forecast_df = pd.DataFrame({
        'Gas': gas,
        'Date': forecast_dates,
        'Forecast': forecast_mean.values,
        'Lower_CI': forecast_ci.iloc[:, 0].values,
        'Upper_CI': forecast_ci.iloc[:, 1].values
    })
    
    all_forecasts.append(forecast_df)

final_forecast_df = pd.concat(all_forecasts, ignore_index=True)
output_csv_path = "ghg_sarima_forecasts_next_6_months.csv"
final_forecast_df.to_csv(output_csv_path, index=False)
print(f"\n✅ Semua prediksi disimpan ke: {output_csv_path}")

# --- 3. Create Interactive 2x2 Panel Chart using HTML ---
print("\n--- Membuat Grafik HTML Interaktif ---")

try:
    with open("logo_bmkg.png", "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()
    logo_source = f"data:image/png;base64,{logo_base64}"
    logo_available = True
except FileNotFoundError:
    logo_source = ""
    logo_available = False

# Define plot properties
gas_colors = {'CO2_seasonal': 'red', 'CH4_seasonal': 'blue', 'N2O_seasonal': 'green', 'SF6_seasonal': 'orange'}
gas_names_id = {'CO2_seasonal': 'CO₂', 'CH4_seasonal': 'CH₄', 'N2O_seasonal': 'N₂O', 'SF6_seasonal': 'SF₆'}
gas_units = {'CO2_seasonal': 'ppm', 'CH4_seasonal': 'ppb', 'N2O_seasonal': 'ppb', 'SF6_seasonal': 'ppt'}
gas_rgba_colors = {'CO2_seasonal': 'rgba(255,0,0,0.2)', 'CH4_seasonal': 'rgba(0,0,255,0.2)', 'N2O_seasonal': 'rgba(0,128,0,0.2)', 'SF6_seasonal': 'rgba(255,165,0,0.2)'}

# Create subplots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[f'{gas_names_id[gas]} ({gas_units[gas]})' for gas in target_gases],
    vertical_spacing=0.18,
    horizontal_spacing=0.10
)

# Filter historical data
last_12_months_start = df.index.max() - pd.DateOffset(months=11)
historical_filtered = df[df.index >= last_12_months_start]

# Define subplot positions
subplot_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

for i, gas in enumerate(target_gases):
    row, col = subplot_positions[i]
    gas_name = gas_names_id[gas]
    color = gas_colors[gas]
    
    # Historical Data
    fig.add_trace(go.Scatter(
        x=historical_filtered.index, y=historical_filtered[gas], mode='lines+markers',
        name=f'Data Historis {gas_name}', marker=dict(size=4, color=color), line=dict(color=color, width=2),
        showlegend=True, legendgroup=f'group{i}',
        hovertemplate=f'<b>{gas_name} Historis</b><br>Tanggal: %{{x|%b %Y}}<br>Konsentrasi: %{{y:.2f}} {gas_units[gas]}<extra></extra>'
    ), row=row, col=col)
    
    # Forecast Data
    gas_forecast = final_forecast_df[final_forecast_df['Gas'] == gas]
    fig.add_trace(go.Scatter(
        x=gas_forecast['Date'], y=gas_forecast['Forecast'], mode='lines+markers',
        name=f'Prediksi {gas_name}', marker=dict(size=6, symbol='star', color=color), line=dict(color=color, width=3, dash='dot'),
        showlegend=True, legendgroup=f'group{i}',
        hovertemplate=f'<b>{gas_name} Prediksi</b><br>Tanggal: %{{x|%b %Y}}<br>Konsentrasi: %{{y:.2f}} {gas_units[gas]}<extra></extra>'
    ), row=row, col=col)

    # Confidence Interval
    fig.add_trace(go.Scatter(
        x=pd.concat([gas_forecast['Date'], gas_forecast['Date'][::-1]]),
        y=pd.concat([gas_forecast['Upper_CI'], gas_forecast['Lower_CI'][::-1]]),
        fill='toself', fillcolor=gas_rgba_colors[gas], line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip", name=f'Interval Kepercayaan 95% {gas_name}', showlegend=True, legendgroup=f'group{i}'
    ), row=row, col=col)

# Add logos if available
images_list = []
if logo_available:
    subplot_refs = [("x domain", "y domain"), ("x2 domain", "y2 domain"), ("x3 domain", "y3 domain"), ("x4 domain", "y4 domain")]
    for xref, yref in subplot_refs:
        images_list.append(dict(source=logo_source, xref=xref, yref=yref, x=0.95, y=0.05, sizex=0.12, sizey=0.12, xanchor="right", yanchor="bottom", opacity=0.6, layer="above"))

# --- MODIFIED: Layout, Font, and Language Updates ---

final_annotations = list(fig.layout.annotations)
for anno in final_annotations:
    anno.font.size = 18

final_annotations.append(dict(
    text="<i></i>", showarrow=False, xref="paper", yref="paper",
    x=0.5, y=-0.22, xanchor='center', yanchor='top',
    font=dict(size=12, color="gray")
))

# Update layout
fig.update_layout(
    #title={
    #    'text': '<b>Prediksi Konsentrasi Gas Rumah Kaca 6 Bulan ke Depan</b>',
    #    'x': 0.5, 'xanchor': 'center', 'font': {'size': 24}
    #},
    template="plotly_white", hovermode="x unified",
    height=900, width=1800,
    margin=dict(l=300, r=80, t=120, b=200),
    legend=dict(
        orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5,
        font=dict(size=13),
        bgcolor="rgba(255,255,255,0.9)", bordercolor="rgba(0,0,0,0.3)", borderwidth=1
    ),
    annotations=final_annotations,
    images=images_list
)

# Update axes
for i, gas in enumerate(target_gases):
    row, col = subplot_positions[i]
    fig.update_xaxes(
        title_text="Tanggal", title_font=dict(size=14), tickfont=dict(size=12),
        gridcolor='lightgrey', tickformat='%b %Y', row=row, col=col
    )
    fig.update_yaxes(
        title_text=f"Konsentrasi ({gas_units[gas]})", title_font=dict(size=14), tickfont=dict(size=12),
        gridcolor='lightgrey', row=row, col=col
    )

# Save to HTML
output_html_path = "index.html"
fig.write_html(
    output_html_path,
    config={'locale': 'id', 'displayModeBar': True, 'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d'], 'displaylogo': False}
)

print(f"✅ Grafik disimpan di: {output_html_path}")
print("\n--- Script Selesai ---")

