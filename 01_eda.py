# Imports
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# Settings
DATA_PATH = "data/train.csv"

STORE_SAMPLE = [1, 85, 262, 550, 897]

OUTPUT_FOLDER = "output/eda"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Loading Data
df = pd.read_csv(DATA_PATH, low_memory=False)
df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Open'] == 1]

print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

print(df.head())

# Basic Summary Stats
print("\nSummary Statistics:")
print(df[['Sales', 'Customers', 'Promo']].describe())
print(f"\nMissing values:\n{df.isnull().sum()}")

# Total Sales Over Time (All Stores)
weekly_sales = (
    df.groupby('Date')['Sales']
    .sum()
    .resample('W')
    .sum()
)

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(weekly_sales.index, weekly_sales.values, color='#2c7bb6', linewidth=1.2)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

plt.xticks(rotation=45)

ax.set_title('Total Weekly Sales — All Stores', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Total Sales (€)')
plt.tight_layout()

plt.savefig(f"{OUTPUT_FOLDER}/01_total_weekly_sales.png", dpi=150)

plt.close()

# Promo Impact on Sales
promo_impact = df.groupby('Promo')['Sales'].mean()

lift_pct = ((promo_impact[1] - promo_impact[0]) / promo_impact[0]) * 100

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(
    ['No Promo', 'Promo'],
    promo_impact.values,
    color=['#d7191c', '#1a9641'],
    width=0.5
)

ax.bar_label(bars, fmt='€%.0f', padding=5)

ax.set_title(f'Average Daily Sales: Promo vs. No Promo\n(+{lift_pct:.1f}% lift)', fontsize=13)
ax.set_ylabel('Average Daily Sales (€)')
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/02_promo_impact.png", dpi=150)
plt.close()
print("Chart saved: promo impact")

# Seasonality - Avg Sales by Month
df['Month'] = df['Date'].dt.month
monthly_avg = df.groupby('Month')['Sales'].mean()

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(month_names, monthly_avg.values, color='#756bb1', width=0.6)
ax.set_title('Average Daily Sales by Month (Seasonality)', fontsize=13, fontweight='bold')
ax.set_ylabel('Average Daily Sales (€)')
ax.set_xlabel('Month')
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/03_seasonality_by_month.png", dpi=150)
plt.close()
print("Chart saved: seasonality by month")

# Store Level Variation
store_avg = df.groupby('Store')['Sales'].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(range(len(store_avg)), store_avg.values, color='#2c7bb6', linewidth=1)

ax.set_title('Store Sales Distribution — Ranked (High to Low)', fontsize=13, fontweight='bold')
ax.set_xlabel('Store Rank')
ax.set_ylabel('Average Daily Sales (€)')
ax.axhline(store_avg.mean(), color='red', linestyle='--', linewidth=1, label=f'Mean: €{store_avg.mean():,.0f}')

ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/04_store_distribution.png", dpi=150)
plt.close()
print("Chart saved: store distribution")

# Sample Store More Info
sample_df = df[df['Store'].isin(STORE_SAMPLE)]

fig, axes = plt.subplots(len(STORE_SAMPLE), 1, figsize=(14, 3 * len(STORE_SAMPLE)), sharex=True)

for i, store_id in enumerate(STORE_SAMPLE):
    store_data = (
        sample_df[sample_df['Store'] == store_id]
        .set_index('Date')['Sales']
        .resample('W').sum()
    )

    axes[i].plot(store_data.index, store_data.values, linewidth=1, color='#2c7bb6')
    axes[i].set_title(f'Store {store_id} — Weekly Sales', fontsize=11)
    axes[i].set_ylabel('Sales (€)')
    axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    axes[i].xaxis.set_major_locator(mdates.MonthLocator(interval=3))

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/05_sample_stores.png", dpi=150)
plt.close()
print("Chart saved: sample stores")

print("\n✓ EDA complete. Charts saved to:", OUTPUT_FOLDER)
print("Next step: open 02_forecasting.py")