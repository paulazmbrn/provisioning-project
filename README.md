# Demand Forecasting & Inventory Optimization
## Beauty Retail Category | Python · Scikit-learn · Pandas · Matplotlib · Tableau

## Business Context
Beauty retail is one of the most operationally demanding categories in consumer goods because demand is seaosnal, promo-driven, and unforgiving. A shelf that runs out two weeks before Christmas does not recover that revenue and a pallet of slow-moving stock sitting in a warehouse after a promotion ends limits working capital with no guarantee of sell-through. Provisioning teams exist to prevent both of these outcomes and translate demand signals into purchase orders that arrive at the right place, in the right quantity, at the right time. This project models that process end-to-end from raw sales history to reorder recommendations using real retail data from a German drugstore chain (Rossmann Stores), the main primary distribution channel for mass-market beauty in most European and Latin American markets.

## Objective
- Forecast weekly demand at the store level using historical sales, promotional activity, and seasonal patterns.
- Identify peak demand periods that require advance inventory positioning
- Generate safety stock and reorder point recommendations per store based on forecast output and a 95% service level target.

## Dataset
**Source:** [ Rossmann Store Sales - Kaggle ] (https://www.kaggle.com/competitions/rossmann-store-sales/overview)

**Scope:** 5 representative stores selected from 1,115 across Germany | ~2.5 years of daily sales history (2013-2015) | variables include sales, customers, promotional activity, store type, and holiday indicators

**Relevance:** Drugstores are the primary retail channel for mass-market beauty brands (skincare, haircare, cosmetics). The demand dynamics in this dataset (seasonal peaks, promo lift, store-level volatility) directly mirror the planning challenges a provisioning team at a beauty company would face.
## Methodology
**Exploratory Data Analysis** — before modeling, the data was explored for seasonal patterns, promotional lift, and store-level variation. December and public holidays emerged as consistent demand drivers. Promotional weeks showed a meaningful lift over baseline across all store types. Store sales volumes varied significantly across the sample, confirming that a single replenishment plan cannot apply uniformly across locations.

**Demand Forecasting** — a linear regression model was built per store using engineered features: cyclical week-of-year encoding (sine/cosine), promotional flag, December and holiday indicators, and lag features capturing sales 4, 8, and 52 weeks prior. The 52-week lag is particularly important — it anchors each prediction to the same period in the prior year, which is the most reliable seasonal signal available. Models were evaluated on a 12-week holdout period with no look-ahead.

**Inventory Logic** — forecast outputs were translated into three operational parameters per store: safety stock (buffer inventory calculated from forecast error volatility at a 95% service level), reorder point (the stock level that triggers a new order, accounting for a 2-week supplier lead time), and weeks of coverage (how long on-hand stock would last at forecast demand). A peak demand threshold of 120% of store average was used to flag weeks requiring advance ordering.

**Assumptions** — supplier lead time is fixed at 2 weeks; service level target is 95% (stockout acceptable in 1 week out of 20); demand is measured in sales value (€) as a proxy for unit volume; promotional calendar is assumed known one cycle in advance.

## Key Findings
- **Forecast accuracy ranges from 6.0% to 14.8% MAPE** across the five stores. Store 550 is the hardest to predict (14.8%), meaning its replenishment plan carries more uncertainty and requires a proportionally larger safety stock buffer than more stable stores like Store 262 (6.0% MAPE).

- **Safety stock requirements vary by nearly 10x across the sample** — from €3,228 (Store 897, low-volume, stable) to €30,221 (Store 262, high-volume, higher absolute error). This means provisioning plans cannot be templated uniformly: each store profile requires its own buffer calculation.

- **All five stores sit at approximately 2.2–2.3 weeks of coverage at reorder point**, which falls below the 3-week minimum target set for this analysis. This is a structural finding: under the current demand and lead time assumptions, every store in the sample would be at stockout risk if a supplier delay occurred. Orders should be placed earlier or minimum stock targets should be revised upward.

- **Store 897 is the only store with detected peak demand weeks**, with average peak demand reaching €21,824 — approximately 21% above its baseline weekly average of €18,007. For this store, inventory should be positioned at least 2 lead-time cycles (4 weeks) in advance of the flagged peak period to avoid stockout during elevated demand.

- **Promotional activity is the single strongest demand driver**, with a regression coefficient of approximately €9,000 per week when a promotion is active. Replenishment plans must treat promotional and baseline demand as separate planning cycles — ordering to a baseline forecast during a promotional week will systematically understock.

## Inventory Recommendations
Calculated at 95% service level · 2-week lead time assumption · values in €

| Store | Avg Weekly Demand | Safety Stock | Reorder Point | Weeks of Coverage | MAPE (%) |
|-------|-------------------|--------------|---------------|-------------------|----------|
| 1     | €25,280           | €4,442       | €55,001       | 2.2               | 6.5%     |
| 85    | €52,437           | €13,940      | €118,815      | 2.3               | 8.3%     |
| 262   | €145,116          | €30,221      | €320,453      | 2.2               | 6.0%     |
| 550   | €37,388           | €12,743      | €87,519       | 2.3               | 14.8%    |
| 897   | €18,007           | €3,228       | €39,241       | 2.2               | 7.3%     |


## Limitations
- This model operates at store level in sales value (€). A production provisioning system at a CPG company would plan at SKU level, by channel and geography, with far greater data granularity and many more variables including supplier reliability, warehouse capacity, and in-market competitive activity.
- Lead time is assumed constant at 2 weeks. In practice, supplier and logistics lead times are variable and themselves need to be modeled — fixed lead time assumptions will underestimate safety stock requirements in volatile supply environments.
- The promotional calendar is assumed known in advance. In reality, replenishment alignment with commercial promo plans requires close cross-functional coordination between supply chain, brand, and sales teams — a process this model simplifies significantly.
  
## Stack
Python (pandas, NumPy, Matplotlib, scikit-learn) · Tableau · GitHub

## About
Built as a portfolio project targeting supply chain and provisioning roles in the consumer goods industry. I am currently transitioning into data analytics from an operational background since I currently work as an Assistant Golf Professional in Los Cabos, México and I am completing a data science degree at Universidad del Valle de México in parallel. This project represents that transition in practice: taking a real business problem, applying Python and machine learning to it, and translating the output into the kind of operational recommendation a team can actually act on.
