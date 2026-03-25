# Imports
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

import os
import warnings
warnings.filterwarnings('ignore')

from scipy import stats


# Settings
FORECAST_PATH  = "output/forecasting/forecast_results.csv"
METRICS_PATH   = "output/forecasting/model_metrics.csv"
OUTPUT_FOLDER  = "output/inventory"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

LEAD_TIME_WEEKS   = 2      # assumed supplier lead time
SERVICE_LEVEL     = 0.95   # 95% — stockout acceptable 1 week in 20
Z_SCORE           = stats.norm.ppf(SERVICE_LEVEL)  # = 1.645
PEAK_THRESHOLD    = 1.20   # flag weeks where forecast > 120% of store average

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

def save_chart(fig, filename):
    fig.savefig(f"{OUTPUT_FOLDER}/{filename}", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {filename}")


# Load Data

print("Loading forecast results...")
df      = pd.read_csv(FORECAST_PATH)
metrics = pd.read_csv(METRICS_PATH)

df['Date'] = pd.to_datetime(df['Date'])

print(f"  Stores: {df['Store'].nunique()}")
print(f"  Weeks per store: {df.groupby('Store').size().iloc[0]}")


# Inventory Calculations Per Store

print("\nCalculating inventory parameters...")

records = []

for store_id in df['Store'].unique():
    sdf = df[df['Store'] == store_id].copy()

    avg_weekly_demand   = sdf['Predicted'].mean()
    std_forecast_error  = sdf['Error'].std()

    # Safety stock: buffer to absorb forecast error at target service level
    # Formula: Z × σ(error) × √(lead time)
    safety_stock = Z_SCORE * std_forecast_error * np.sqrt(LEAD_TIME_WEEKS)
    safety_stock = max(safety_stock, 0)

    # Reorder point: stock level at which a new order should be placed
    # Formula: avg demand × lead time + safety stock
    reorder_point = (avg_weekly_demand * LEAD_TIME_WEEKS) + safety_stock

    # Peak demand: weeks where forecast exceeds store average by threshold
    peak_weeks      = sdf[sdf['Predicted'] > avg_weekly_demand * PEAK_THRESHOLD]
    peak_demand_avg = peak_weeks['Predicted'].mean() if len(peak_weeks) > 0 else avg_weekly_demand

    # Weeks of coverage at average demand if holding reorder point stock
    weeks_coverage = reorder_point / avg_weekly_demand if avg_weekly_demand > 0 else 0

    # Pull MAPE from metrics for this store
    mape = metrics.loc[metrics['Store'] == store_id, 'MAPE (%)'].values[0]

    records.append({
        'Store':              store_id,
        'Avg Weekly Demand':  round(avg_weekly_demand, 0),
        'Forecast Error Std': round(std_forecast_error, 0),
        'Safety Stock':       round(safety_stock, 0),
        'Reorder Point':      round(reorder_point, 0),
        'Peak Demand (avg)':  round(peak_demand_avg, 0),
        'Peak Weeks Count':   len(peak_weeks),
        'Weeks of Coverage':  round(weeks_coverage, 1),
        'MAPE (%)':           mape,
    })

inv_df = pd.DataFrame(records)

print("\n--- Inventory Recommendations ---")
print(inv_df.to_string(index=False))


# Chart - Safety Stock and Reorder Point by Store

fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor(PALETTE["bg"])

x      = np.arange(len(inv_df))
width  = 0.35

bars1 = ax.bar(x - width/2, inv_df['Safety Stock'],   width, color=PALETTE["accent"],  label='Safety Stock',  zorder=3)
bars2 = ax.bar(x + width/2, inv_df['Reorder Point'],  width, color=PALETTE["primary"], label='Reorder Point', zorder=3, alpha=0.85)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            fmt_euros(bar.get_height()), ha='center', va='bottom', fontsize=8)

for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            fmt_euros(bar.get_height()), ha='center', va='bottom', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels([f"Store {s}" for s in inv_df['Store']])
ax.yaxis.set_major_formatter(euro_formatter)
ax.set_title("Safety Stock & Reorder Point by Store", loc='left')
ax.text(0.0, 1.02,
        f"Service level: {SERVICE_LEVEL*100:.0f}%  ·  Lead time assumption: {LEAD_TIME_WEEKS} weeks  ·  Values in €",
        transform=ax.transAxes, fontsize=8.5, color=PALETTE["neutral"], ha='left')
ax.legend()
plt.tight_layout()
save_chart(fig, "01_safety_stock_reorder.png")


# Chart - Peak Demand Weeks by Store

fig, axes = plt.subplots(len(inv_df), 1, figsize=(13, 3.8 * len(inv_df)))
fig.patch.set_facecolor(PALETTE["bg"])

for i, store_id in enumerate(inv_df['Store']):
    ax  = axes[i]
    sdf = df[df['Store'] == store_id].copy()
    avg = sdf['Predicted'].mean()

    is_peak = sdf['Predicted'] > avg * PEAK_THRESHOLD

    ax.fill_between(sdf['Date'], sdf['Predicted'],
                    alpha=0.15, color=PALETTE["neutral"])
    ax.plot(sdf['Date'], sdf['Predicted'],
            color=PALETTE["neutral"], linewidth=1.2, label='Forecast')
    ax.fill_between(sdf['Date'], sdf['Predicted'],
                    where=is_peak,
                    color=PALETTE["negative"], alpha=0.45, label='Peak risk weeks')
    ax.axhline(avg, color=PALETTE["accent"], linewidth=1.2,
               linestyle='--', label=f'Store avg: {fmt_euros(avg)}')
    ax.axhline(avg * PEAK_THRESHOLD, color=PALETTE["negative"], linewidth=0.8,
               linestyle=':', label=f'Peak threshold (+20%): {fmt_euros(avg * PEAK_THRESHOLD)}')

    ax.yaxis.set_major_formatter(euro_formatter)
    ax.set_title(f"Store {store_id}", fontsize=11, fontweight='bold', loc='left', pad=16)
    ax.text(0.0, 1.03,
            f"Peak weeks: {is_peak.sum()}  ·  Avg peak demand: {fmt_euros(sdf.loc[is_peak, 'Predicted'].mean() if is_peak.any() else 0)}  ·  Reorder point: {fmt_euros(inv_df.loc[inv_df['Store']==store_id, 'Reorder Point'].values[0])}",
            transform=ax.transAxes, fontsize=8.5, color=PALETTE["neutral"], ha='left')

    if i == 0:
        ax.legend(loc='upper right', fontsize=8)

fig.suptitle("Forecast Demand with Peak Risk Periods Highlighted",
             fontsize=14, fontweight='bold', y=1.005)
fig.subplots_adjust(hspace=0.55)
save_chart(fig, "02_peak_demand_weeks.png")


# Chart - Weeks of Coverage vs. Target

TARGET_COVERAGE = 3.0

fig, ax = plt.subplots(figsize=(9, 4.5))
fig.patch.set_facecolor(PALETTE["bg"])

colors = [PALETTE["positive"] if w >= TARGET_COVERAGE else PALETTE["negative"]
          for w in inv_df['Weeks of Coverage']]

bars = ax.barh(
    [f"Store {s}" for s in inv_df['Store']],
    inv_df['Weeks of Coverage'],
    color=colors, height=0.5, zorder=3
)
ax.axvline(TARGET_COVERAGE, color=PALETTE["accent"], linewidth=1.5,
           linestyle='--', label=f'Target: {TARGET_COVERAGE} weeks')

for bar, val in zip(bars, inv_df['Weeks of Coverage']):
    ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
            f"{val:.1f} wks", va='center', fontsize=9)

ax.set_title("Weeks of Coverage at Reorder Point", loc='left')
ax.text(0.0, 1.02,
        "Green = meets target  ·  Red = below target  ·  Target set at 3 weeks minimum",
        transform=ax.transAxes, fontsize=8.5, color=PALETTE["neutral"], ha='left')
ax.set_xlabel("Weeks of Coverage")
ax.legend()
plt.tight_layout()
save_chart(fig, "03_weeks_coverage.png")


# Export

output_csv = f"{OUTPUT_FOLDER}/inventory_recommendations.csv"
inv_df.to_csv(output_csv, index=False)
print(f"\n  ✓ Inventory recommendations exported: {output_csv}")
print("     → Load into Tableau alongside forecast_results.csv")
print("\n✓ Inventory complete.")
