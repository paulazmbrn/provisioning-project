# Imports
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
# Accuracy metrics:
# MAE = Mean Absolute Error: on average, how many € off are we?
# MAPE = Mean Absolute Percentage Error: how far off as a %?
# MAPE is more useful here because stores have very different
# sales volumes — % error lets you compare across them fairly.

from sklearn.model_selection import TimeSeriesSplit

# Settings
DATA_PATH = "data/train.csv"
STORE_SAMPLE = [1, 85, 262, 550, 897]

OUTPUT_FOLDER = "output/forecasting"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


PALETTE = {
    "primary":   "#2C3E7A",   
    "accent":    "#E8A838",   
    "positive":  "#3A8C5C",   
    "negative":  "#C0392B",   
    "neutral":   "#7F8C8D",  
    "bg":        "#FAFAF8",  
    "grid":      "#EDEDE8",   
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["bg"],
    "axes.edgecolor":    PALETTE["neutral"],
    "axes.labelcolor":   "#2C2C2C",
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.titlepad":     14,
    "axes.labelsize":    10,
    "axes.grid":         True,
    "grid.color":        PALETTE["grid"],
    "grid.linewidth":    0.8,
    "xtick.color":       "#555555",
    "ytick.color":       "#555555",
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "legend.framealpha": 0.85,
    "font.family":       "sans-serif",
})

# Helper Functions

def fmt_euros(x, pos=None):
    if abs(x) >= 1_000_000:
        return f"€{x/1_000_000:.1f}M"
    elif abs(x) >= 1_000:
        return f"€{x/1_000:.0f}K"
    else:
        return f"€{x:.0f}"

euro_formatter = mticker.FuncFormatter(fmt_euros)

def add_subtitle(ax, text):
    """Adds a small grey subtitle line below the chart title."""
    ax.annotate(
        text,
        xy=(0.0, 1.02), xycoords='axes fraction',
        fontsize=8.5, color=PALETTE["neutral"],
        ha='left', va='bottom'
    )

def save_chart(fig, filename):
    fig.savefig(f"{OUTPUT_FOLDER}/{filename}", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {filename}")

# Load and Prep Data
print("Loading data...")
df = pd.read_csv(DATA_PATH, low_memory=False)
df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Open'] == 1].copy()

# Extract time features from the date
df['Year']        = df['Date'].dt.year
df['Month']       = df['Date'].dt.month
df['Week']        = df['Date'].dt.isocalendar().week.astype(int)
df['DayOfWeek']   = df['Date'].dt.dayofweek

df['IsDecember']  = (df['Month'] == 12).astype(int)
df['IsHoliday']   = (df['StateHoliday'] != '0').astype(int)

print(f"  Rows loaded: {len(df):,}")
print(f"  Stores: {df['Store'].nunique()}")
print(f"  Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")

# Feature Engineering
def build_features(store_df):
    # --- Aggregate to weekly level ---
    weekly = (
        store_df
        .set_index('Date')
        .resample('W')
        .agg({
            'Sales':       'sum',     
            'Customers':   'sum',    
            'Promo':       'max',     
            'IsDecember':  'max',     
            'IsHoliday':   'max',     
        })
        .dropna()
    )

    weekly['Week'] = weekly.index.isocalendar().week.astype(int)
    weekly['Year'] = weekly.index.year

    weekly['week_sin'] = np.sin(2 * np.pi * weekly['Week'] / 52)
    weekly['week_cos'] = np.cos(2 * np.pi * weekly['Week'] / 52)

    weekly['lag_4']  = weekly['Sales'].shift(4)   # 4 weeks ago
    weekly['lag_8']  = weekly['Sales'].shift(8)   # 8 weeks ago
    weekly['lag_52'] = weekly['Sales'].shift(52)  # same week last year

    weekly['roll_4'] = weekly['Sales'].shift(1).rolling(4).mean()

    weekly = weekly.dropna()

    feature_cols = [
        'Promo', 'IsDecember', 'IsHoliday',
        'week_sin', 'week_cos',
        'lag_4', 'lag_8', 'lag_52',
        'roll_4'
    ]

    X = weekly[feature_cols]
    y = weekly['Sales']

    return X, y, weekly

# Train / Evaluate Model Per Store
print("\nTraining models...")

all_results   = []   # will hold forecast vs actual rows for every store
model_metrics = []   # will hold accuracy stats per store

for store_id in STORE_SAMPLE:
    print(f"  Store {store_id}...")

    store_df = df[df['Store'] == store_id].copy()
    X, y, weekly = build_features(store_df)

    # Time-aware train/test split
    split_idx = len(X) - 12
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred_test  = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    y_pred_test  = np.clip(y_pred_test, 0, None)
    y_pred_train = np.clip(y_pred_train, 0, None)

    # Calculate Accuracy
    mae  = mean_absolute_error(y_test, y_pred_test)
    mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100

    model_metrics.append({
        'Store': store_id,
        'MAE (€)': round(mae, 0),
        'MAPE (%)': round(mape, 1),
        'Test Weeks': len(y_test)
    })

    # Storing results for charts
    test_dates = weekly.index[split_idx:]
    for date, actual, predicted in zip(test_dates, y_test, y_pred_test):
        all_results.append({
            'Store':     store_id,
            'Date':      date,
            'Actual':    actual,
            'Predicted': predicted,
            'Error':     predicted - actual
        })

    # Print accuracy summary
    metrics_df = pd.DataFrame(model_metrics)
    print("\n--- Model Accuracy (Test Period) ---")
    print(metrics_df.to_string(index=False))


# Chart - Forecast vs. Actual by Store

print("\nGenerating charts...")

results_df = pd.DataFrame(all_results)

fig, axes = plt.subplots(
    len(STORE_SAMPLE), 1,
    figsize=(14, 4.5 * len(STORE_SAMPLE)),  # slightly taller per panel
)
fig.patch.set_facecolor(PALETTE["bg"])

for i, store_id in enumerate(STORE_SAMPLE):
    ax = axes[i]
    sdf = results_df[results_df['Store'] == store_id]
    metrics_row = metrics_df[metrics_df['Store'] == store_id].iloc[0]

    ax.fill_between(
        sdf['Date'], sdf['Actual'],
        alpha=0.18, color=PALETTE["neutral"], label='Actual'
    )
    ax.plot(sdf['Date'], sdf['Actual'],
            color=PALETTE["neutral"], linewidth=1.2, label='_nolegend_')
    ax.plot(sdf['Date'], sdf['Predicted'],
            color=PALETTE["accent"], linewidth=2,
            linestyle='--', label='Forecast', zorder=5)

    # --- Euro formatting on y-axis for EVERY subplot ---
    ax.yaxis.set_major_formatter(euro_formatter)

    # --- Title: left-aligned store label ---
    ax.set_title(f"Store {store_id}", fontsize=11, fontweight='bold', loc='left', pad=18)
    # pad=18 lifts the title up enough to leave room for the subtitle below it.

    # --- Subtitle: sits just below the title ---
    ax.text(
        0.0, 1.04,
        f"MAPE: {metrics_row['MAPE (%)']:.1f}%  ·  MAE: {fmt_euros(metrics_row['MAE (€)'])}  ·  12-week holdout",
        transform=ax.transAxes,
        fontsize=8.5, color=PALETTE["neutral"],
        ha='left', va='bottom'
    )

    if i < len(STORE_SAMPLE) - 1:
        ax.set_xticklabels([])
        # Hide x-tick labels on all but the bottom chart.
        # sharex=True was removed above so each panel can control this independently.
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.get_xticklabels(), rotation=35, ha='right')

    if i == 0:
        ax.legend(loc='upper right', fontsize=9)

    fig.suptitle(
    "Forecast vs. Actual — Weekly Sales (12-Week Holdout)",
    fontsize=14, fontweight='bold', y=1.005
    )

    fig.subplots_adjust(hspace=0.55)

    save_chart(fig, "01_forecast_vs_actual.png")

# Chart - Forecast Error Distribution
fig, axes = plt.subplots(1, len(STORE_SAMPLE), figsize=(15, 5))
fig.patch.set_facecolor(PALETTE["bg"])

for i, store_id in enumerate(STORE_SAMPLE):
    ax = axes[i]
    sdf = results_df[results_df['Store'] == store_id]

    errors = sdf['Error']
    mean_err = errors.mean()
    color = PALETTE["positive"] if mean_err >= 0 else PALETTE["negative"]

    ax.hist(errors, bins=8, color=color, alpha=0.75, edgecolor='white')
    ax.axvline(0, color='black', linewidth=1, linestyle='--', label='Zero error')
    ax.axvline(mean_err, color=PALETTE["accent"], linewidth=1.5,
               linestyle=':', label=f'Mean: {fmt_euros(mean_err)}')

    ax.xaxis.set_major_formatter(euro_formatter)
    ax.set_title(f"Store {store_id}", fontsize=10, fontweight='bold', pad=10)
    ax.set_xlabel("Error\n(Predicted − Actual)", fontsize=8.5)


    if i == 0:
        ax.set_ylabel("Number of Weeks", fontsize=9)

    ax.legend(fontsize=7.5, loc='upper right')


fig.suptitle(
    "Forecast Error Distribution by Store",
    fontsize=13, fontweight='bold', y=1.06
)


fig.text(
    0.5, 1.01,
    "Errors near zero = unbiased  ·  Negative mean = stockout risk",
    ha='center', va='bottom',
    fontsize=8.5, color=PALETTE["neutral"]
)
fig.text(
    0.5, 0.975,
    "Positive mean = overstock risk  ·  Color: green = over-predict, red = under-predict",
    ha='center', va='bottom',
    fontsize=8.5, color=PALETTE["neutral"]
)


fig.subplots_adjust(wspace=0.45, top=0.88)


save_chart(fig, "02_error_distribution.png")

# Chart -  Feature Importance
coef_records = []
feature_cols = ['Promo', 'IsDecember', 'IsHoliday',
                'week_sin', 'week_cos',
                'lag_4', 'lag_8', 'lag_52', 'roll_4']

for store_id in STORE_SAMPLE:
    store_df = df[df['Store'] == store_id].copy()
    X, y, _ = build_features(store_df)
    model = LinearRegression().fit(X, y)
    for feat, coef in zip(feature_cols, model.coef_):
        coef_records.append({'Feature': feat, 'Coefficient': coef})

coef_df = (
    pd.DataFrame(coef_records)
    .groupby('Feature')['Coefficient']
    .mean()
    .sort_values()
)

label_map = {
    'Promo':       'Promotion active',
    'IsDecember':  'December indicator',
    'IsHoliday':   'Public holiday',
    'week_sin':    'Week-of-year (sin)',
    'week_cos':    'Week-of-year (cos)',
    'lag_4':       'Sales 4 weeks ago',
    'lag_8':       'Sales 8 weeks ago',
    'lag_52':      'Sales 52 weeks ago (YoY)',
    'roll_4':      '4-week rolling avg',
}
coef_df.index = [label_map.get(i, i) for i in coef_df.index]

colors = [PALETTE["positive"] if v >= 0 else PALETTE["negative"] for v in coef_df.values]

fig, ax = plt.subplots(figsize=(11, 5.5))
fig.patch.set_facecolor(PALETTE["bg"])

bars = ax.barh(coef_df.index, coef_df.values, color=colors, height=0.6)
ax.axvline(0, color='black', linewidth=0.8)

# --- Value labels on each bar ---
for bar, val in zip(bars, coef_df.values):
    label = fmt_euros(val)
    x_pos = val + (max(coef_df.values) * 0.01) if val >= 0 else val - (max(coef_df.values) * 0.01)
    ha = 'left' if val >= 0 else 'right'
    ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
            label, va='center', ha=ha, fontsize=8, color='#2C2C2C')

ax.xaxis.set_major_formatter(euro_formatter)
ax.set_title("Feature Importance — Average Regression Coefficients",
             loc='left', pad=22)


ax.text(0.0, 1.055,
        "Positive = drives sales higher  ·  Negative = associated with lower sales",
        transform=ax.transAxes, fontsize=8.5, color=PALETTE["neutral"], ha='left')
ax.text(0.0, 1.02,
        "Averaged across sample stores  ·  Lag features near zero: model relies primarily on calendar signals",
        transform=ax.transAxes, fontsize=8.5, color=PALETTE["neutral"], ha='left')

ax.set_xlabel("Average Coefficient Value (€ impact per unit of feature)", labelpad=10)
plt.tight_layout()
save_chart(fig, "03_feature_importance.png")

label_map = {
    'Promo':       'Promotion active',
    'IsDecember':  'December indicator',
    'IsHoliday':   'Public holiday',
    'week_sin':    'Week-of-year (sin)',
    'week_cos':    'Week-of-year (cos)',
    'lag_4':       'Sales 4 weeks ago',
    'lag_8':       'Sales 8 weeks ago',
    'lag_52':      'Sales 52 weeks ago (YoY)',
    'roll_4':      '4-week rolling avg',
}
coef_df.index = [label_map.get(i, i) for i in coef_df.index]

colors = [PALETTE["positive"] if v >= 0 else PALETTE["negative"] for v in coef_df.values]

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(PALETTE["bg"])
bars = ax.barh(coef_df.index, coef_df.values, color=colors, height=0.6)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title("Feature Importance — Average Regression Coefficients", loc='left')
add_subtitle(ax, "Positive = drives sales higher  ·  Negative = associated with lower sales  ·  Averaged across sample stores")
ax.set_xlabel("Average Coefficient Value (€ impact per unit of feature)")
ax.xaxis.set_major_formatter(euro_formatter)
plt.tight_layout()
save_chart(fig, "03_feature_importance.png")

# Export Results
results_df['Error_Pct'] = (
    (results_df['Predicted'] - results_df['Actual']) / results_df['Actual'] * 100
).round(1)

output_csv = f"{OUTPUT_FOLDER}/forecast_results.csv"
results_df.to_csv(output_csv, index=False)
print(f"\n  ✓ Forecast results exported: {output_csv}")
print("     → Load this file into Tableau for the dashboard")

metrics_df.to_csv(f"{OUTPUT_FOLDER}/model_metrics.csv", index=False)
print(f"  ✓ Model metrics exported: model_metrics.csv")