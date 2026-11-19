# E-commerce Sales and Customer Behaviour: Exploratory Data Analysis

**Track 2: Data Science** | Python, Pandas, NumPy, Matplotlib, Seaborn, SciPy

Exploratory and statistical analysis of three years (19 Mar 2012 to 19 Mar 2015) of website sessions, pageviews, orders, order items and refunds for an online plush-toy retailer. The goal is to understand customer behaviour, sales patterns and business performance, and to turn the findings into recommendations.

## Key findings
- **Growth:** gross revenue grew 204% in 2013 and 174% in 2014, driven by both more traffic and a higher conversion rate (4.1% in 2012 to 7.2% in 2014).
- **Mobile gap:** mobile brings 31% of sessions but 14% of orders. It converts at 3.1% versus 8.5% on desktop, with a similar order value.
- **Cross-selling:** a second-item option lifted average order value from $49.99 to $63.80.
- **A/B tests:** in a concurrent test, the new billing page completed 62.1% of orders versus 45.1% for the old one.
- **Refunds:** 4.3% of units are refunded. A Mr. Fuzzy refund spike in Aug-Sep 2014 cost about $10K.
- **Retention:** only 1.9% of customers ever place a second order.

## What the analysis covers
| Task | Content |
|---|---|
| 1. Data exploration | Loading, shapes and data types, missing values and duplicates, relationships between tables, cleaning |
| 2. Exploratory analysis | Revenue over time, best-selling products, purchasing behaviour, marketing channels, mobile vs desktop, session-to-order conversion and funnel, refunds, new vs returning visitors |
| 3. Statistical analysis | Descriptive statistics, traffic vs orders correlation and regression, period comparisons, product sales vs refunds, 14 hypothesis tests, 13 visualisations |
| 4. Recommendations | Seven evidence-based business insights with what-if sensitivities |

## Repository structure
```
.
├── Ecommerce_EDA_Notebook.ipynb     # Full analysis with markdown and commented code
├── ecommerce_eda.py                 # Same analysis as a Python script
├── EDA_Report_Ecommerce_Sales_and_Customer_Behaviour.pdf
├── Video_Presentation_Script.md     # Script for the video walkthrough
├── figures/                         # 13 saved charts (PNG)
├── tables/                          # Summary tables (CSV) and key_metrics.json
└── data/                            # Place the six CSV files here
```

## Datasets
Place these six files in the `data/` folder (not included in this repository):
`website_sessions.csv`, `website_pageviews.csv`, `orders_mav.csv`, `order_items.csv`, `order_item_refunds.csv`, `products.csv`

## How to run
1. Install the requirements:
   ```
   pip install pandas numpy matplotlib seaborn scipy jupyter
   ```
2. Tell the code where the data is (default is the current folder):
   ```
   # macOS / Linux
   export DATA_DIR=data
   # Windows (PowerShell)
   $env:DATA_DIR = "data"
   ```
3. Run the notebook (`jupyter notebook Ecommerce_EDA_Notebook.ipynb`) or the script (`python ecommerce_eda.py`). Charts are saved to `figures/` and tables to `tables/`.

## Data notes and limitations
- **Truncated pageviews file:** `website_pageviews.csv` contains exactly Excel's row limit (1,048,575 rows) and stops on 16 Jan 2015. Page-level results (funnel, landing pages, billing test) therefore use the 89% of sessions with complete pageview data. All other results use the full data.
- **Partial years:** 2012 (from 19 March) and 2015 (to 19 March) are incomplete, so trends use complete months and like-for-like windows.
- **No cost or customer data:** advertising costs and demographics are not available, so return on ad spend and customer segmentation are not covered.
- **Observational data:** apart from the concurrent A/B tests, results show association, not causation. The what-if values are sensitivities based on 2014 volumes, not forecasts.

