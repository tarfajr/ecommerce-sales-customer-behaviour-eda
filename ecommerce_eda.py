"""
Exploratory Data Analysis of E-commerce Sales and Customer Behaviour (Track 2: Data Science)
Python source version of the notebook. Place the six CSV files next to this script
(or set the DATA_DIR environment variable) and run:  python ecommerce_eda.py
"""

# ==============================================================================
# # Exploratory Data Analysis of E-commerce Sales and Customer Behaviour
# Track 2: Data Science | Prepared by: Solomon Buba Tarfa | Tools: Python, Pandas, NumPy, Matplotlib, Seaborn, SciPy
# ## Business objective
# Investigate customer behaviour, sales patterns and business performance using statistical and exploratory data analysis techniques, for an online retailer of plush toys ("Fuzzy Factory") that sells four products.
# ## Datasets used
# | File | Grain (one row =) | Key columns |
# ==============================================================================

# ==============================================================================
# ## 0. Setup
# Import the libraries, set display options and define small helper functions used throughout the notebook.
# ==============================================================================

# Import the operating-system module to build file paths that work on any computer
import os
# Import warnings so that harmless library warnings do not clutter the output
import warnings
# Import json so key results can be saved to a file at the end
import json
# Import pandas, the main library for tabular data
import pandas as pd
# Import NumPy for fast numerical operations
import numpy as np
# Import Matplotlib's plotting interface to draw charts
import matplotlib.pyplot as plt
# Import Seaborn for attractive statistical charts built on Matplotlib
import seaborn as sns
# Import the SciPy statistics module for hypothesis tests and correlations
from scipy import stats

# Use IPython's display() to show tables nicely in Jupyter/Colab
try:
    # Try to import the rich display function
    from IPython.display import display
# If IPython is not installed (plain Python script), fall back to print
except ImportError:
    # print is a simple replacement for display
    display = print

# Folder containing the six CSV files ("." means the same folder as this notebook)
DATA_DIR = os.environ.get("DATA_DIR", ".")
# Folder where every chart will be saved as a PNG image
FIG_DIR = os.environ.get("FIG_DIR", "figures")
# Create the chart folder if it does not exist yet
os.makedirs(FIG_DIR, exist_ok=True)

# Hide non-critical warnings to keep the output clean
warnings.filterwarnings("ignore")
# Apply a clean, light Seaborn theme to every chart
sns.set_theme(style="whitegrid", context="notebook")
# Set a sharp resolution for charts shown in the notebook
plt.rcParams["figure.dpi"] = 110
# Remove the top and right chart borders for a modern look
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
# Show numbers in pandas tables with thousands separators and 2 decimals
pd.set_option("display.float_format", "{:,.2f}".format)
# Allow pandas tables to use a wider screen area
pd.set_option("display.width", 170)
# Show up to 30 columns before pandas truncates a table
pd.set_option("display.max_columns", 30)

# Consistent colour palette used for products and categories in every chart
PALETTE = ["#1F4E79", "#2E86AB", "#F18F01", "#C73E1D", "#6A994E", "#7B2D8E", "#8D99AE"]
# Dictionary that will hold the key numbers quoted in the final report
M = {}
# Helper that formats a number as US dollars without decimals
money = lambda x: f"${x:,.0f}"
# Helper that formats a fraction (0.0683) as a percentage string ("6.8%")
pct = lambda x, d=1: f"{x * 100:.{d}f}%"

def save_fig(name):
    """Tidy the layout, save the current figure as a PNG and display it."""
    # Adjust spacing so that labels and titles are not clipped
    plt.tight_layout()
    # Save the chart into the figures folder at print quality
    plt.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=150, bbox_inches="tight")
    # Display the chart in the notebook
    plt.show()

# ==============================================================================
# # Task 1: Data Exploration
# ## 1.1 Load the datasets into Python
# All created_at columns are stored as text in the format month/day/year hour:minute, so we parse them into real datetime values while loading.
# ==============================================================================

# Map a short dataset name to its CSV file name
FILES = {
    "sessions": "website_sessions.csv",
    "pageviews": "website_pageviews.csv",
    "orders": "orders_mav.csv",
    "order_items": "order_items.csv",
    "refunds": "order_item_refunds.csv",
    "products": "products.csv",
}
# Date format used by every created_at column (month/day/year hour:minute)
DATE_FMT = "%m/%d/%Y %H:%M"

# Read every CSV file into a dictionary of DataFrames
raw = {name: pd.read_csv(os.path.join(DATA_DIR, file)) for name, file in FILES.items()}
# Loop over each dataset to convert its text timestamps into datetime values
for name, frame in raw.items():
    # Parse the created_at text using the known format
    frame["created_at"] = pd.to_datetime(frame["created_at"], format=DATE_FMT)

# Unpack the dictionary into separate, readable variable names
sessions, pageviews, orders = raw["sessions"], raw["pageviews"], raw["orders"]
# Unpack the remaining three tables
order_items, refunds, products = raw["order_items"], raw["refunds"], raw["products"]

# Print how many rows were loaded for every dataset
for name, frame in raw.items():
    # Show the dataset name and its row count with thousands separators
    print(f"{name:<12} -> {len(frame):>9,} rows loaded")

# ==============================================================================
# ## 1.2 Examine the shape, columns and data types
# ==============================================================================

# Create an empty list that will collect one summary row per dataset
shape_rows = []
# Loop over every dataset
for name, frame in raw.items():
    # Store the dataset name, row count, column count, date range and memory footprint
    shape_rows.append({
        "dataset": name,
        "rows": len(frame),
        "columns": frame.shape[1],
        "first_record": frame["created_at"].min(),
        "last_record": frame["created_at"].max(),
        "memory_mb": frame.memory_usage(deep=True).sum() / 1e6,
    })
# Convert the list into a DataFrame and display it
shape_summary = pd.DataFrame(shape_rows)
# Show the shape summary
display(shape_summary)

# Create an empty list that will collect one summary row per column
schema_rows = []
# Loop over every dataset
for name, frame in raw.items():
    # Loop over every column of the dataset
    for col in frame.columns:
        # Record the column's data type, number of non-null values and number of distinct values
        schema_rows.append({
            "dataset": name,
            "column": col,
            "dtype": str(frame[col].dtype),
            "non_null": int(frame[col].notna().sum()),
            "unique_values": int(frame[col].nunique()),
        })
# Convert the list into a DataFrame
schema = pd.DataFrame(schema_rows)
# Display the full schema (columns and data types of all six datasets)
display(schema)

# ==============================================================================
# Observations
# - The six tables describe one funnel: *sessions -> pageviews -> orders -> order items -> refunds*, plus a small product lookup table.
# - Every created_at column is now a proper datetime64 value, and all identifiers are integers.
# - Sessions, orders and order items cover 19 Mar 2012 to 19 Mar 2015 (three years). Refunds run a little longer because they happen *after* the order.
# ==============================================================================

# ==============================================================================
# ## 1.3 Identify missing values and duplicate records
# ==============================================================================

# Count missing values in every column of every dataset and stack the results
missing = pd.concat({name: frame.isna().sum() for name, frame in raw.items()}, names=["dataset", "column"])
# Turn the result into a tidy table with a readable column name
missing = missing.rename("missing_values").reset_index()
# Keep only the columns that actually contain missing values
missing = missing[missing["missing_values"] > 0].copy()
# Compute the percentage of rows affected in each column
missing["pct_of_rows"] = missing.apply(lambda r: r["missing_values"] / len(raw[r["dataset"]]) * 100, axis=1)
# Display the columns with missing values
display(missing)

# Define the primary key (unique identifier) of every dataset
PRIMARY_KEYS = {
    "sessions": "website_session_id", "pageviews": "website_pageview_id", "orders": "order_id",
    "order_items": "order_item_id", "refunds": "order_item_refund_id", "products": "product_id",
}
# Create an empty list for the duplicate-check results
dup_rows = []
# Loop over every dataset and its primary key
for name, key in PRIMARY_KEYS.items():
    # Count fully identical rows and repeated primary-key values
    dup_rows.append({
        "dataset": name,
        "fully_duplicated_rows": int(raw[name].duplicated().sum()),
        "duplicated_primary_keys": int(raw[name][key].duplicated().sum()),
    })
# Display the duplicate-check table
display(pd.DataFrame(dup_rows))

# Flag the sessions that have no UTM marketing tags
no_utm = sessions["utm_source"].isna()
# Flag the sessions that have no HTTP referrer
no_referer = sessions["http_referer"].isna()
# Cross-tabulate the two flags to understand WHY values are missing
print(pd.crosstab(no_utm.rename("utm_missing"), no_referer.rename("referer_missing")))
# Confirm that utm_source, utm_campaign and utm_content are always missing together
print("\nUTM columns missing together:", bool((sessions[["utm_source", "utm_campaign", "utm_content"]].isna().all(axis=1) == no_utm).all()))

# ==============================================================================
# Interpretation of missing values
# - Missing values appear only in the marketing columns of website_sessions. They are *structural*, not data-entry errors:
# - UTM tags missing and a search-engine referrer present = organic search traffic (visitor arrived from a search engine without a paid-ad tag).
# - UTM tags missing and no referrer = direct traffic (typed URL / bookmark).
# - There are no duplicate rows and no duplicate primary keys in any dataset.
# - We therefore do not drop or impute these rows. Instead, in Section 1.5 we convert them into explicit, meaningful categories.
# ==============================================================================

# ==============================================================================
# ### Data-completeness check: is website_pageviews.csv complete?
# The pageviews file has a suspicious row count. We test whether it was cut off.
# ==============================================================================

# Number of data rows in the pageviews file
pv_rows = len(pageviews)
# Highest session id that appears in the pageviews file
PV_MAX_SESSION = int(pageviews["website_session_id"].max())
# Highest session id that appears in the sessions file
SESSION_MAX = int(sessions["website_session_id"].max())
# Excel's hard limit is 1,048,576 rows including the header, i.e. 1,048,575 data rows
EXCEL_DATA_ROW_LIMIT = 1_048_576 - 1

# Report whether the row count equals the Excel limit
print(f"Pageview rows: {pv_rows:,}  | Excel data-row limit: {EXCEL_DATA_ROW_LIMIT:,}  | identical: {pv_rows == EXCEL_DATA_ROW_LIMIT}")
# Report the last session and last timestamp covered by pageviews
print(f"Last session in pageviews: {PV_MAX_SESSION:,} of {SESSION_MAX:,} total sessions")
# Report the last pageview date versus the last session date
print(f"Last pageview: {pageviews['created_at'].max()}  | Last session: {sessions['created_at'].max()}")
# Compute the share of sessions fully covered by the pageview data
PV_COVERAGE = (sessions["website_session_id"] < PV_MAX_SESSION).mean()
# Print the coverage percentage
print(f"Share of sessions with complete pageview data: {pct(PV_COVERAGE)}")
# Store the results for the report
M.update(pv_rows=pv_rows, pv_max_session=PV_MAX_SESSION, session_max=SESSION_MAX,
         pv_cutoff_date=pageviews["created_at"].max(), pv_coverage=PV_COVERAGE)

# ==============================================================================
# Important data limitation. The pageviews file contains exactly Excel's maximum number of rows, and it ends on {...} (session {...} of {...}). It was almost certainly truncated when it was exported through Excel. Therefore:
# - All analyses that use sessions, orders, items and refunds are complete (full 3 years).
# - Only the page-level funnel and landing-page analysis is restricted to the {...} of sessions whose pageviews are fully present (up to mid-January 2015). This is still a very large sample (400,000+ sessions) so conclusions remain reliable, but it is flagged wherever it applies.
# ==============================================================================

# ==============================================================================
# ## 1.4 Investigate the relationships between the datasets
# The tables form a star-shaped relational model:
# products (product_id) ---< order_items (product_id)
# sessions (website_session_id) ---< pageviews (website_session_id)
# sessions (website_session_id) ---- orders (website_session_id)     [1 order max per session]
# orders (order_id) ---< order_items (order_id)
# ==============================================================================

# Each tuple: (child table.column, parent table.column, number of child rows with no matching parent)
integrity = [
    ("orders.website_session_id", "sessions.website_session_id", (~orders["website_session_id"].isin(sessions["website_session_id"])).sum()),
    ("pageviews.website_session_id", "sessions.website_session_id", (~pageviews["website_session_id"].isin(sessions["website_session_id"])).sum()),
    ("order_items.order_id", "orders.order_id", (~order_items["order_id"].isin(orders["order_id"])).sum()),
    ("order_items.product_id", "products.product_id", (~order_items["product_id"].isin(products["product_id"])).sum()),
    ("refunds.order_item_id", "order_items.order_item_id", (~refunds["order_item_id"].isin(order_items["order_item_id"])).sum()),
    ("refunds.order_id", "orders.order_id", (~refunds["order_id"].isin(orders["order_id"])).sum()),
]
# Convert the checks to a DataFrame (0 orphan rows means a perfect relationship)
integrity = pd.DataFrame(integrity, columns=["child_column", "parent_column", "orphan_rows"])
# Display the referential-integrity results
display(integrity)

# Add up the item lines of every order (count, price and cost) from order_items
item_totals = order_items.groupby("order_id").agg(items=("order_item_id", "count"), price=("price_usd", "sum"), cogs=("cogs_usd", "sum"))
# Attach those totals to the orders table for comparison
check = orders.set_index("order_id").join(item_totals)
# Verify that the item count per order equals items_purchased
print("items_purchased matches item lines :", bool((check["items_purchased"] == check["items"]).all()))
# Verify that order price equals the sum of its item prices
print("order price = sum of item prices   :", bool(np.isclose(check["price_usd"], check["price"]).all()))
# Verify that order cost equals the sum of its item costs
print("order cogs  = sum of item costs    :", bool(np.isclose(check["cogs_usd"], check["cogs"]).all()))
# Verify that every order has exactly one primary item
print("exactly one primary item per order :", bool((order_items.groupby("order_id")["is_primary_item"].sum() == 1).all()))
# Verify that a session produced at most one order
print("at most one order per session      :", bool(orders["website_session_id"].is_unique))
# Verify that no refund is larger than the price of the item it refunds
refund_check = refunds.merge(order_items[["order_item_id", "price_usd"]], on="order_item_id")
# Print the result of the refund-amount check
print("refund amount <= item price        :", bool((refund_check["refund_amount_usd"] <= refund_check["price_usd"] + 1e-9).all()))
# Verify that every item is refunded at most once
print("at most one refund per item        :", bool(refunds["order_item_id"].is_unique))

# ==============================================================================
# Conclusion. All relationships are intact: there are zero orphan records, order totals reconcile exactly with their item lines, and each session produces at most one order. The datasets can be joined safely.
# ==============================================================================

# ==============================================================================
# ## 1.5 Clean and prepare the data for analysis
# Cleaning steps:
# 1. Make the missing marketing values explicit and build a single marketing channel column.
# 2. Add time features (year, quarter, month, weekday, hour).
# 3. Enrich sessions with a conversion flag and revenue.
# 4. Enrich order items with product names, refunds, profit and net revenue.
# ==============================================================================

# Work on copies so the raw tables stay untouched
sessions = raw["sessions"].copy()
# Replace missing UTM tags with the explicit label "none"
for col in ["utm_source", "utm_campaign", "utm_content"]:
    # Fill the missing values of this column
    sessions[col] = sessions[col].fillna("none")
# Replace a missing referrer with the explicit label "none"
sessions["http_referer"] = sessions["http_referer"].fillna("none")

# Conditions that define each marketing channel (evaluated top to bottom)
channel_conditions = [
    (sessions["utm_source"] == "none") & (sessions["http_referer"] == "none"),
    (sessions["utm_source"] == "none"),
    (sessions["utm_source"] == "gsearch") & (sessions["utm_campaign"] == "nonbrand"),
    (sessions["utm_source"] == "bsearch") & (sessions["utm_campaign"] == "nonbrand"),
    (sessions["utm_campaign"] == "brand"),
    (sessions["utm_source"] == "socialbook"),
]
# Human-readable channel names matching the conditions above
channel_names = ["Direct", "Organic search", "Google nonbrand", "Bing nonbrand", "Paid brand search", "Social (Socialbook)"]
# Create the channel column; anything unmatched would be labelled "Other"
sessions["channel"] = np.select(channel_conditions, channel_names, default="Other")
# Make sure that no session was left unclassified
print("Unclassified sessions:", int((sessions["channel"] == "Other").sum()))
# Readable device and visitor-type labels
sessions["device_type"] = sessions["device_type"].str.title()
# Convert the 0/1 repeat flag into a readable label
sessions["visitor_type"] = np.where(sessions["is_repeat_session"] == 1, "Returning", "New")

def add_time_features(df):
    """Add calendar columns derived from created_at and return the DataFrame."""
    # Calendar date without the time part
    df["date"] = df["created_at"].dt.normalize()
    # Calendar year
    df["year"] = df["created_at"].dt.year
    # Quarter label such as "2014Q3"
    df["quarter"] = df["created_at"].dt.to_period("Q").astype(str)
    # First day of the month (used for monthly grouping)
    df["month"] = df["created_at"].dt.to_period("M").dt.to_timestamp()
    # Weekday number (Monday = 0)
    df["weekday_num"] = df["created_at"].dt.dayofweek
    # Weekday name
    df["weekday"] = df["created_at"].dt.day_name()
    # Hour of the day (0-23)
    df["hour"] = df["created_at"].dt.hour
    # Return the enriched table
    return df

# Work on copies of the other tables as well
orders = raw["orders"].copy()
# Copy the order items table
order_items = raw["order_items"].copy()
# Copy the refunds table
refunds = raw["refunds"].copy()
# Add time features to the four time-stamped business tables
sessions, orders, order_items, refunds = [add_time_features(t) for t in (sessions, orders, order_items, refunds)]

# Short, chart-friendly product names
SHORT_NAMES = {
    "The Original Mr. Fuzzy": "Mr. Fuzzy", "The Forever Love Bear": "Love Bear",
    "The Birthday Sugar Panda": "Sugar Panda", "The Hudson River Mini bear": "Mini Bear",
}
# Add the short name to the products table
products["short_name"] = products["product_name"].map(SHORT_NAMES)

# Total refunded dollars per order item (an item can be refunded at most once)
refund_by_item = refunds.groupby("order_item_id")["refund_amount_usd"].sum()
# Attach product names to the order items
items = order_items.merge(products[["product_id", "product_name", "short_name"]], on="product_id", how="left")
# Refund amount of each item (0 when never refunded)
items["refund_usd"] = items["order_item_id"].map(refund_by_item).fillna(0)
# 1 if the item was refunded, else 0
items["is_refunded"] = (items["refund_usd"] > 0).astype(int)
# Gross profit before refunds = price minus cost of goods sold
items["gross_profit_usd"] = items["price_usd"] - items["cogs_usd"]
# Revenue after subtracting refunds
items["net_revenue_usd"] = items["price_usd"] - items["refund_usd"]
# Profit after refunds (refunded money is lost; product cost is not recovered)
items["net_profit_usd"] = items["gross_profit_usd"] - items["refund_usd"]
# Label each item as the primary item or an added cross-sell item
items["item_role"] = np.where(items["is_primary_item"] == 1, "Primary item", "Cross-sell item")

# Refund dollars and refunded item count per order
order_refunds = items.groupby("order_id").agg(order_refund_usd=("refund_usd", "sum"), refunded_items=("is_refunded", "sum"))
# Join the refund totals onto the orders table
orders = orders.merge(order_refunds, on="order_id", how="left")
# Gross profit of each order before refunds
orders["gross_profit_usd"] = orders["price_usd"] - orders["cogs_usd"]
# Net revenue of each order after refunds
orders["net_revenue_usd"] = orders["price_usd"] - orders["order_refund_usd"]
# Flag orders that contain two items (a bundle / cross-sell order)
orders["is_bundle"] = (orders["items_purchased"] == 2).astype(int)
# Attach channel, device and visitor type of the session that produced each order
orders = orders.merge(sessions[["website_session_id", "channel", "device_type", "visitor_type", "utm_content"]], on="website_session_id", how="left")

# Map each session id to the revenue of its order (sessions without an order get 0)
session_revenue = orders.set_index("website_session_id")["price_usd"]
# Add a 0/1 conversion flag to every session
sessions["converted"] = sessions["website_session_id"].isin(orders["website_session_id"]).astype(int)
# Add the revenue generated by every session
sessions["revenue_usd"] = sessions["website_session_id"].map(session_revenue).fillna(0)

# Sanity check 1: item-level revenue must equal order-level revenue
assert np.isclose(items["price_usd"].sum(), orders["price_usd"].sum()), "Revenue mismatch"
# Sanity check 2: no missing values in the engineered columns
assert orders[["channel", "device_type", "visitor_type"]].notna().all().all(), "Missing session attributes"
# Sanity check 3: prices and costs must all be positive
assert (items[["price_usd", "cogs_usd"]] > 0).all().all(), "Non-positive price or cost found"

# Define the first and last COMPLETE calendar months (data starts 19 Mar 2012 and ends 19 Mar 2015)
FIRST_FULL_MONTH, LAST_FULL_MONTH = pd.Timestamp("2012-04-01"), pd.Timestamp("2015-02-01")
# Define the first and last COMPLETE days to be used for daily analyses
FIRST_FULL_DAY, LAST_FULL_DAY = sessions["date"].min() + pd.Timedelta(days=1), sessions["date"].max() - pd.Timedelta(days=1)
# Print a short summary of the cleaned data
print(f"Sessions: {len(sessions):,} | Orders: {len(orders):,} | Order items: {len(items):,} | Refunds: {len(refunds):,}")
# Print the headline totals
print(f"Gross revenue: {money(orders['price_usd'].sum())} | Refunds: {money(refunds['refund_amount_usd'].sum())} | Gross profit: {money(orders['gross_profit_usd'].sum())}")
# Show the cleaned sessions table
display(sessions.head(3))

# ==============================================================================
# Data is clean and ready. Cleaning outcomes:
# - No rows were removed (no duplicates or invalid values were found); missing marketing fields were converted into six explicit channels.
# - 2012 and 2015 are partial years (data starts 19 Mar 2012 and ends 19 Mar 2015). To avoid misleading comparisons we use only complete months/days for trends and use *like-for-like* windows for year-over-year comparisons.
# ==============================================================================

# ==============================================================================
# # Task 2: Exploratory Data Analysis
# Each sub-section answers one of the eight business questions with a table, a chart and an interpretation.
# ## 2.1 How does revenue change over time?
# *Revenue* here means gross revenue (order price before refunds). Refunds are shown separately so that net revenue is also available.
# ==============================================================================

# Aggregate orders into calendar months
monthly = orders.groupby("month").agg(
    orders=("order_id", "count"), gross_revenue=("price_usd", "sum"),
    cogs=("cogs_usd", "sum"), refunds=("order_refund_usd", "sum")).reset_index()
# Net revenue = gross revenue minus refunds
monthly["net_revenue"] = monthly["gross_revenue"] - monthly["refunds"]
# Gross profit = gross revenue minus cost of goods sold
monthly["gross_profit"] = monthly["gross_revenue"] - monthly["cogs"]
# Average order value (revenue per order)
monthly["aov"] = monthly["gross_revenue"] / monthly["orders"]
# Mark the months that are complete (the first and last months are partial)
monthly["complete"] = monthly["month"].between(FIRST_FULL_MONTH, LAST_FULL_MONTH)
# Revenue of complete months only (partial months become NaN)
complete_rev = monthly["gross_revenue"].where(monthly["complete"])
# Month-over-month growth rate in percent (complete months only)
monthly["mom_growth_pct"] = complete_rev.pct_change() * 100
# 3-month moving average to smooth short-term noise
monthly["revenue_ma3"] = complete_rev.rolling(3).mean()

# Aggregate orders into calendar quarters
qtr = orders.groupby("quarter").agg(orders=("order_id", "count"), gross_revenue=("price_usd", "sum"), gross_profit=("gross_profit_usd", "sum")).reset_index()
# A quarter is complete unless it is the first (2012Q1) or last (2015Q1) partial quarter
qtr["complete"] = ~qtr["quarter"].isin(["2012Q1", "2015Q1"])
# Year-over-year growth versus the same quarter one year earlier (only between complete quarters)
qtr["yoy_growth_pct"] = (qtr["gross_revenue"].pct_change(4) * 100).where(qtr["complete"] & qtr["complete"].shift(4, fill_value=False))

# Annual summary: sessions and conversion rate by year
yearly_sessions = sessions.groupby("year").agg(sessions=("website_session_id", "count"), converted=("converted", "sum"))
# Annual summary: revenue metrics by year
yearly = orders.groupby("year").agg(orders=("order_id", "count"), gross_revenue=("price_usd", "sum"),
    refunds=("order_refund_usd", "sum"), gross_profit=("gross_profit_usd", "sum"), aov=("price_usd", "mean"))
# Combine the two annual tables
yearly = yearly.join(yearly_sessions)
# Add the session-to-order conversion rate
yearly["conversion_rate_pct"] = yearly["converted"] / yearly["sessions"] * 100
# Add the gross margin percentage
yearly["gross_margin_pct"] = yearly["gross_profit"] / yearly["gross_revenue"] * 100
# Describe the coverage of each year (2012 and 2015 are partial)
yearly["coverage"] = ["partial (from 19 Mar)", "full year", "full year", "partial (to 19 Mar)"]
# Display the annual summary
display(yearly.drop(columns="converted"))

# Display the quarterly summary (complete quarters have a YoY growth rate)
display(qtr)
# Select the best complete month by gross revenue
best = monthly[monthly["complete"]].sort_values("gross_revenue", ascending=False).iloc[0]
# Store headline numbers for the report
M.update(
    rev_total=orders["price_usd"].sum(), refunds_total=refunds["refund_amount_usd"].sum(), gp_total=orders["gross_profit_usd"].sum(),
    orders_total=len(orders), sessions_total=len(sessions), aov_total=orders["price_usd"].mean(),
    rev_2012=yearly.loc[2012, "gross_revenue"], rev_2013=yearly.loc[2013, "gross_revenue"],
    rev_2014=yearly.loc[2014, "gross_revenue"], rev_2015=yearly.loc[2015, "gross_revenue"],
    growth_13=(yearly.loc[2013, "gross_revenue"] / yearly.loc[2012, "gross_revenue"] - 1),
    growth_14=(yearly.loc[2014, "gross_revenue"] / yearly.loc[2013, "gross_revenue"] - 1),
    cvr_2012=yearly.loc[2012, "conversion_rate_pct"], cvr_2014=yearly.loc[2014, "conversion_rate_pct"], cvr_2015=yearly.loc[2015, "conversion_rate_pct"],
    sessions_2013=yearly.loc[2013, "sessions"], sessions_2014=yearly.loc[2014, "sessions"],
    aov_2012=yearly.loc[2012, "aov"], aov_2014=yearly.loc[2014, "aov"],
    best_month=best["month"].strftime("%B %Y"), best_month_rev=best["gross_revenue"],
    margin_total=orders["gross_profit_usd"].sum() / orders["price_usd"].sum(),
)

# Create a wide figure for the revenue trend
fig, ax = plt.subplots(figsize=(12, 5.5))
# Split the monthly table into complete and partial months
comp, part = monthly[monthly["complete"]], monthly[~monthly["complete"]]
# Draw monthly revenue (in thousands of dollars) for the complete months
ax.plot(comp["month"], comp["gross_revenue"] / 1000, color=PALETTE[0], marker="o", ms=4, lw=1.8, label="Monthly gross revenue")
# Draw the smoothed 3-month moving average
ax.plot(comp["month"], comp["revenue_ma3"] / 1000, color=PALETTE[2], lw=2.8, label="3-month moving average")
# Show the two partial months as hollow markers
ax.scatter(part["month"], part["gross_revenue"] / 1000, facecolor="white", edgecolor=PALETTE[0], zorder=5, label="Partial month (data starts/ends 19 Mar)")
# Mark each product launch after the first product with a dashed vertical line
for _, row in products.iloc[1:].iterrows():
    # Draw the vertical launch line
    ax.axvline(row["created_at"].normalize(), ls="--", color="grey", lw=1)
    # Label the line with the product name
    ax.text(row["created_at"].normalize(), ax.get_ylim()[1] * 0.6, f" {row['short_name']} launch", rotation=90, va="bottom", fontsize=8, color="dimgrey")
# Format the y-axis as thousands of dollars
ax.yaxis.set_major_formatter(lambda x, p: f"${x:,.0f}K")
# Add title and axis labels
ax.set(title="Monthly gross revenue grew steadily, with strong end-of-year peaks", xlabel="", ylabel="Gross revenue")
# Show the legend in the upper-left corner
ax.legend(loc="upper left", frameon=False)
# Save and show the figure
save_fig("fig01_monthly_revenue_trend")

# ==============================================================================
# Findings - revenue over time
# - Gross revenue grew from {...} (partial 2012, from 19 March) to {...} in 2013 ({...}) and {...} in 2014 ({...}). The business has earned {...} in total at a {...} gross margin.
# - The growth is steady rather than a one-off jump, and the best complete month was {...} ({...}).
# - Revenue peaks every November-December (holiday season) and softens in January (down versus December in all three years), which signals clear seasonality (quantified in Task 3).
# - Growth came from two sources at once: more traffic (sessions rose from {...} in 2013 to {...} in 2014) and a better conversion rate ({...}% in 2012 to {...}% in 2014).
# - Average order value rose from {...} to {...}, helped by the cross-sell feature (see 2.3).
# ==============================================================================

# ==============================================================================
# ## 2.2 Which products generate the highest sales?
# ==============================================================================

# Aggregate order items by product
prod = items.groupby("product_name").agg(
    units=("order_item_id", "count"), revenue=("price_usd", "sum"), cogs=("cogs_usd", "sum"),
    refunded_units=("is_refunded", "sum"), refunds=("refund_usd", "sum")).reset_index()
# Bring in the short name and launch date of each product
prod = prod.merge(products[["product_name", "short_name", "created_at"]].rename(columns={"created_at": "launch_date"}), on="product_name")
# Share of total revenue contributed by each product
prod["revenue_share_pct"] = prod["revenue"] / prod["revenue"].sum() * 100
# Gross profit and gross margin percentage per product
prod["gross_profit"] = prod["revenue"] - prod["cogs"]
# Gross margin as a percentage of revenue
prod["gross_margin_pct"] = prod["gross_profit"] / prod["revenue"] * 100
# Average selling price per unit
prod["avg_price"] = prod["revenue"] / prod["units"]
# Refund rate = refunded units / units sold
prod["refund_rate_pct"] = prod["refunded_units"] / prod["units"] * 100
# Profit after subtracting refunds
prod["net_profit"] = prod["gross_profit"] - prod["refunds"]
# Number of days each product has been on sale until the end of the data
prod["days_on_sale"] = (sessions["created_at"].max() - prod["launch_date"]).dt.days
# Revenue earned per day on sale (fair comparison between old and new products)
prod["revenue_per_day"] = prod["revenue"] / prod["days_on_sale"]
# Sort by revenue so the best seller comes first
prod = prod.sort_values("revenue", ascending=False).reset_index(drop=True)
# Display the product performance table
display(prod[["product_name", "launch_date", "units", "revenue", "revenue_share_pct", "avg_price", "gross_margin_pct", "net_profit", "revenue_per_day", "refund_rate_pct"]])

# Revenue per quarter and product (rows = quarter, columns = product), 0 before launch
prod_q = items.pivot_table(index="quarter", columns="short_name", values="price_usd", aggfunc="sum").fillna(0)
# Order the columns by launch date of the products
prod_q = prod_q[products["short_name"].tolist()]
# Create two side-by-side charts
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1, 1.6]})
# Left chart: total revenue by product as horizontal bars (best seller on top)
bars = axes[0].barh(prod["short_name"][::-1], prod["revenue"][::-1] / 1000, color=PALETTE[:4][::-1])
# Write the revenue and its share inside each bar label
for bar, (_, r) in zip(bars, prod.iloc[::-1].iterrows()):
    # Place the label just to the right of each bar
    axes[0].text(bar.get_width() + 15, bar.get_y() + bar.get_height() / 2, f"${r['revenue']/1000:,.0f}K ({r['revenue_share_pct']:.0f}%)", va="center", fontsize=9)
# Add a title and axis label to the left chart
axes[0].set(title="Total revenue by product", xlabel="Gross revenue ($ thousands)", xlim=(0, prod["revenue"].max() / 1000 * 1.35))
# Right chart: quarterly revenue stacked by product
prod_q.div(1000).plot(kind="bar", stacked=True, ax=axes[1], color=PALETTE[:4], width=0.85)
# Add a title and axis labels to the right chart
axes[1].set(title="Quarterly revenue by product (new products reshape the mix)", xlabel="", ylabel="Gross revenue ($ thousands)")
# Rotate the quarter labels for readability
axes[1].tick_params(axis="x", rotation=60)
# Place the legend with a clear title
axes[1].legend(title="Product", frameon=False)
# Save and show the figure
save_fig("fig02_revenue_by_product")

# Store the product headline numbers
top = prod.iloc[0]
# Share of revenue of the best seller in the first full quarter of 2013 versus the latest full quarter (2015Q1 is partial, so use 2014Q4)
share_mix = prod_q.div(prod_q.sum(axis=1), axis=0) * 100
# Save the key figures
M.update(top_product=top["product_name"], top_share=top["revenue_share_pct"], top_rev=top["revenue"], top_units=top["units"],
         fuzzy_share_2012q4=share_mix.loc["2012Q4", "Mr. Fuzzy"], fuzzy_share_2014q4=share_mix.loc["2014Q4", "Mr. Fuzzy"],
         lovebear_rev=prod.set_index("short_name").loc["Love Bear", "revenue"], panda_rev=prod.set_index("short_name").loc["Sugar Panda", "revenue"],
         mini_rev=prod.set_index("short_name").loc["Mini Bear", "revenue"], mini_margin=prod.set_index("short_name").loc["Mini Bear", "gross_margin_pct"],
         lovebear_margin=prod.set_index("short_name").loc["Love Bear", "gross_margin_pct"], fuzzy_margin=prod.set_index("short_name").loc["Mr. Fuzzy", "gross_margin_pct"],
         panda_margin=prod.set_index("short_name").loc["Sugar Panda", "gross_margin_pct"])
# Display the product mix (% of quarterly revenue) for the last complete quarters
display(share_mix.round(1).tail(6))

# ==============================================================================
# Findings - best-selling products
# - {...} is the clear leader: {...} in revenue ({...} of the total) and {...} units. It is also the oldest product, on sale for the full three years.
# - The newer products have grown the catalogue: The Forever Love Bear ({...}), The Birthday Sugar Panda ({...}) and The Hudson River Mini bear ({...}). They launched later, so their totals understate their strength (the table shows revenue per day on sale).
# - Mr. Fuzzy was the only product in 2012 ({...}% of revenue); by 2014Q4 its share was {...}%, so the business is less dependent on a single product than it used to be.
# - Gross margins are healthy for every product (Mr. Fuzzy {...}%, Love Bear {...}%, Sugar Panda {...}%, Mini Bear {...}%), and the two newest products, Sugar Panda and Mini Bear, carry the best margins. The Mini Bear is also the cheapest product, which makes it an ideal low-friction add-on (see 2.3).
# ==============================================================================

# ==============================================================================
# ## 2.3 What are the patterns in customer purchasing behaviour?
# We look at (a) basket size and cross-selling, (b) when customers buy (weekday and hour), and (c) repeat buying.
# ==============================================================================

# Date of the first order that contained two items (introduction of the cross-sell feature)
first_bundle = orders.loc[orders["is_bundle"] == 1, "created_at"].min()
# Print the cross-sell launch date
print("First 2-item order:", first_bundle.date())
# Share of 2-item orders and average order value by year
basket_year = orders.groupby("year").agg(orders=("order_id", "count"), bundle_share_pct=("is_bundle", lambda s: s.mean() * 100), aov=("price_usd", "mean"), items_per_order=("items_purchased", "mean"))
# Display the yearly basket table
display(basket_year)
# Average order value of one-item versus two-item orders after cross-selling started
aov_by_size = orders[orders["created_at"] >= first_bundle].groupby("items_purchased")["price_usd"].agg(orders="count", avg_order_value="mean")
# Display the basket-size comparison
display(aov_by_size)

# Map product ids to short product names for readability
id_to_name = products.set_index("product_id")["short_name"]
# Cross-sell attach rate by primary product, using the period after cross-sell launched
post = orders[orders["created_at"] >= first_bundle].copy()
# Translate the primary product id into a name
post["primary_product"] = post["primary_product_id"].map(id_to_name)
# Orders and bundle rate by primary product
attach = post.groupby("primary_product").agg(orders=("order_id", "count"), cross_sell_rate_pct=("is_bundle", lambda s: s.mean() * 100)).sort_values("cross_sell_rate_pct", ascending=False)
# Which products are chosen as the second (cross-sell) item
cross_items = items[items["item_role"] == "Cross-sell item"].groupby("short_name").size().rename("times_added_as_cross_sell").sort_values(ascending=False)
# Display both tables side by side as one result
display(attach)
# Display the cross-sell item popularity
display(cross_items)

# Count orders for every weekday and hour combination
heat = orders.pivot_table(index="weekday", columns="hour", values="order_id", aggfunc="count").fillna(0)
# Put the weekdays in calendar order
heat = heat.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
# Draw the heatmap
fig, ax = plt.subplots(figsize=(13, 4.6))
# Use a sequential colour palette; darker = more orders
sns.heatmap(heat, cmap="Blues", ax=ax, cbar_kws={"label": "Number of orders (all years)"}, linewidths=0.3)
# Add title and axis labels
ax.set(title="When do customers buy? Orders by weekday and hour of day", xlabel="Hour of day (0-23)", ylabel="")
# Save and show the figure
save_fig("fig03_orders_weekday_hour_heatmap")

# Average orders per calendar day for weekdays versus weekend days (complete days only)
daily_orders = orders[(orders["date"] >= FIRST_FULL_DAY) & (orders["date"] <= LAST_FULL_DAY)].groupby("date").size()
# Add the weekday number to each day
dow = daily_orders.groupby(daily_orders.index.dayofweek).mean()
# Average orders on a weekday (Mon-Fri) and a weekend day (Sat-Sun)
wk_avg, we_avg = dow.loc[0:4].mean(), dow.loc[5:6].mean()
# Share of all orders placed between 09:00 and 17:59
office_share = orders["hour"].between(9, 17).mean()
# Share of all orders placed from 00:00 to 06:59 (night)
night_share = orders["hour"].between(0, 6).mean()
# Print the time-pattern results
print(f"Avg orders per weekday: {wk_avg:.1f} | per weekend day: {we_avg:.1f} | weekend is {we_avg / wk_avg * 100:.0f}% of a weekday")
# Print the share of orders in office hours and at night
print(f"Orders between 09:00-17:59: {pct(office_share)} | between 00:00-06:59: {pct(night_share)}")
# Store the results
M.update(wk_avg=wk_avg, we_avg=we_avg, office_share=office_share, night_share=night_share)

# Number of orders placed by each customer (user)
orders_per_user = orders.groupby("user_id").size()
# Number of customers who bought at least twice
repeat_buyers = int((orders_per_user > 1).sum())
# Repeat-buyer share among all customers who ever ordered
repeat_buyer_rate = repeat_buyers / len(orders_per_user)
# Sort orders per customer chronologically
o_sorted = orders.sort_values(["user_id", "created_at"]).copy()
# Time of the previous order of the same customer
o_sorted["previous_order"] = o_sorted.groupby("user_id")["created_at"].shift()
# Days between consecutive orders of the same customer
gap_days = (o_sorted["created_at"] - o_sorted["previous_order"]).dt.days.dropna()
# Share of orders that were placed during a returning-visitor session
orders_on_return_visit = (orders["visitor_type"] == "Returning").mean()
# Print the repeat-purchase statistics
print(f"Customers who ordered: {len(orders_per_user):,} | repeat buyers (2+ orders): {repeat_buyers:,} ({pct(repeat_buyer_rate)})")
# Print timing and return-visit statistics
print(f"Median days between two orders of the same customer: {gap_days.median():.0f}")
# Print how many orders are placed on a return visit
print(f"Orders placed on a return visit (not the first session): {pct(orders_on_return_visit)}")
# Store the results
M.update(customers=len(orders_per_user), repeat_buyers=repeat_buyers, repeat_buyer_rate=repeat_buyer_rate, repeat_gap_median=gap_days.median(), orders_on_return_visit=orders_on_return_visit)

# Quarterly share of 2-item orders and quarterly average order value (complete quarters only)
bq = orders.groupby("quarter").agg(bundle_share=("is_bundle", "mean"), aov=("price_usd", "mean"))
# Remove the two partial quarters
bq = bq.loc[qtr.loc[qtr["complete"], "quarter"]]
# Create two charts side by side
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
# Left chart: share of orders containing two items
axes[0].plot(bq.index, bq["bundle_share"] * 100, marker="o", color=PALETTE[3], lw=2)
# Add the title and labels of the left chart
axes[0].set(title="Cross-selling: share of orders with 2 items", ylabel="% of orders", xlabel="")
# Right chart: average order value
axes[1].bar(bq.index, bq["aov"], color=PALETTE[1])
# Add the title and labels of the right chart
axes[1].set(title="Average order value rises with cross-selling", ylabel="Average order value ($)", xlabel="")
# Rotate the quarter labels on both charts
for ax in axes:
    # Rotate the labels for readability
    ax.tick_params(axis="x", rotation=60)
# Save and show the figure
save_fig("fig04_cross_sell_and_aov")

# Store the basket results used in the narrative
M.update(aov_single=aov_by_size.loc[1, "avg_order_value"], aov_double=aov_by_size.loc[2, "avg_order_value"], first_bundle=first_bundle, bundle_share_2014=basket_year.loc[2014, "bundle_share_pct"] / 100, aov_2013=basket_year.loc[2013, "aov"],
         attach_best=attach.index[0], attach_best_rate=attach.iloc[0]["cross_sell_rate_pct"] / 100,
         attach_worst=attach.index[-1], attach_worst_rate=attach.iloc[-1]["cross_sell_rate_pct"] / 100,
         cross_top=cross_items.index[0], cross_top_n=int(cross_items.iloc[0]))

# ==============================================================================
# Findings - customer purchasing behaviour
# - Cross-selling works. A second-item option appeared on {...}. By 2014, {...} of orders contained two items, lifting the average order value from {...} (2012) to {...} (2014): a two-item order is worth {...} on average versus {...} for a single item. The best cross-sell host is {...} ({...} of its orders add a second item) and the most added second item is {...}.
# - Customers shop during the working day. {...} of orders are placed between 09:00 and 17:59, only {...} overnight. A weekday brings about {...} orders against {...} on a weekend day.
# - Buyers rarely come back to buy again. Only {...} of customers ({...} of {...}) placed a second order; those who did waited a median of {...} days.
# - But many buyers need more than one visit. {...} of orders are placed during a *returning-visitor* session, which means the first visit was research and the purchase happened later.
# ==============================================================================

# ==============================================================================
# ## 2.4 Which marketing channels attract the most website sessions?
# Channels were derived in Section 1.5 from the UTM tags and the referrer.
# ==============================================================================

# Aggregate sessions, orders and revenue by marketing channel
channel = sessions.groupby("channel").agg(
    sessions=("website_session_id", "count"), orders=("converted", "sum"), revenue=("revenue_usd", "sum"),
    first_session=("created_at", "min")).reset_index()
# Share of all sessions coming from each channel
channel["share_of_sessions_pct"] = channel["sessions"] / channel["sessions"].sum() * 100
# Conversion rate = orders / sessions
channel["conversion_rate_pct"] = channel["orders"] / channel["sessions"] * 100
# Revenue earned per session (a measure of traffic quality)
channel["revenue_per_session"] = channel["revenue"] / channel["sessions"]
# Share of all revenue coming from each channel
channel["share_of_revenue_pct"] = channel["revenue"] / channel["revenue"].sum() * 100
# Sort by number of sessions
channel = channel.sort_values("sessions", ascending=False).reset_index(drop=True)
# Display the channel table
display(channel)

# Quarterly sessions by channel (rows = quarter, columns = channel)
ch_q = sessions.pivot_table(index="quarter", columns="channel", values="website_session_id", aggfunc="count").fillna(0)
# Order the columns from the largest to the smallest channel
ch_q = ch_q[channel["channel"].tolist()]
# Create two charts side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
# Left chart: stacked bars of quarterly sessions by channel
ch_q.div(1000).plot(kind="bar", stacked=True, ax=axes[0], color=PALETTE[:6], width=0.85)
# Add title and axis labels to the left chart
axes[0].set(title="Sessions by marketing channel", xlabel="", ylabel="Sessions (thousands)")
# Rotate the quarter labels for readability
axes[0].tick_params(axis="x", rotation=60)
# Move the legend outside to keep the bars readable
axes[0].legend(title="Channel", frameon=False, fontsize=8)
# Right chart: conversion rate per channel as horizontal bars
sorted_ch = channel.sort_values("conversion_rate_pct")
# Draw the bars
axes[1].barh(sorted_ch["channel"], sorted_ch["conversion_rate_pct"], color=PALETTE[1])
# Write the percentage at the end of each bar
for i, v in enumerate(sorted_ch["conversion_rate_pct"]):
    # Place the text label slightly to the right of the bar end
    axes[1].text(v + 0.1, i, f"{v:.1f}%", va="center", fontsize=9)
# Add title and axis label to the right chart
axes[1].set(title="Session-to-order conversion rate by channel", xlabel="Conversion rate (%)", xlim=(0, sorted_ch["conversion_rate_pct"].max() * 1.2))
# Save and show the figure
save_fig("fig05_channels_sessions_and_conversion")

# Look at the paid ad variants (utm_content) to see which creatives and sources perform
ads = sessions[sessions["utm_content"] != "none"].groupby(["utm_source", "utm_campaign", "utm_content"]).agg(
    sessions=("website_session_id", "count"), orders=("converted", "sum"), first_seen=("created_at", "min"), last_seen=("created_at", "max")).reset_index()
# Conversion rate of each ad variant
ads["conversion_rate_pct"] = ads["orders"] / ads["sessions"] * 100
# Display the ad variants sorted by traffic
display(ads.sort_values("sessions", ascending=False))

# Index the channel table by name for easy lookup
ch = channel.set_index("channel")
# Store the channel numbers used in the narrative
M.update(g_nb_sessions=ch.loc["Google nonbrand", "sessions"], g_nb_share=ch.loc["Google nonbrand", "share_of_sessions_pct"] / 100,
         g_nb_rev_share=ch.loc["Google nonbrand", "share_of_revenue_pct"] / 100, g_nb_cvr=ch.loc["Google nonbrand", "conversion_rate_pct"],
         b_nb_cvr=ch.loc["Bing nonbrand", "conversion_rate_pct"], brand_cvr=ch.loc["Paid brand search", "conversion_rate_pct"],
         org_cvr=ch.loc["Organic search", "conversion_rate_pct"], direct_cvr=ch.loc["Direct", "conversion_rate_pct"],
         social_cvr=ch.loc["Social (Socialbook)", "conversion_rate_pct"], social_sessions=ch.loc["Social (Socialbook)", "sessions"],
         social_rps=ch.loc["Social (Socialbook)", "revenue_per_session"], g_nb_rps=ch.loc["Google nonbrand", "revenue_per_session"],
         brand_rps=ch.loc["Paid brand search", "revenue_per_session"], social_first=ch.loc["Social (Socialbook)", "first_session"],
         unpaid_share=(ch.loc[["Direct", "Organic search", "Paid brand search"], "sessions"].sum() / ch["sessions"].sum()))
# Sessions per year and channel to see which channels are growing fastest
ch_year = sessions.pivot_table(index="year", columns="channel", values="website_session_id", aggfunc="count")
# Brand-aware channels = direct + organic + paid brand search
aware_cols = ["Direct", "Organic search", "Paid brand search"]
# Share of yearly sessions that comes from brand-aware channels
aware_share = ch_year[aware_cols].sum(axis=1) / ch_year.sum(axis=1)
# Growth multiple between 2012 and 2014 for brand-aware sessions and for Google non-brand sessions
M.update(aware_share_2012=aware_share[2012], aware_share_2014=aware_share[2014],
         aware_growth=ch_year.loc[2014, aware_cols].sum() / ch_year.loc[2012, aware_cols].sum(),
         gnb_growth=ch_year.loc[2014, "Google nonbrand"] / ch_year.loc[2012, "Google nonbrand"])
# Display the yearly sessions by channel
display(ch_year)

# ==============================================================================
# Findings - marketing channels
# - Google non-brand search is the engine of traffic: {...} sessions, {...} of all visits and {...} of revenue. Bing non-brand is a distant second.
# - Quality differs from quantity. Visitors who already know the brand convert best (paid brand search {...}%, organic {...}%, direct {...}%) while generic non-brand search converts at {...}% (Google) and {...}% (Bing).
# - Socialbook is the weakest channel: it appeared in {...}, delivered {...} sessions but only {...}% conversion and {...} revenue per session (vs. {...} for Google non-brand).
# - Brand-aware traffic is the fastest-growing source. Direct, organic and paid-brand sessions grew {...}x between 2012 and 2014 (Google non-brand: {...}x), lifting their share of all sessions from {...} to {...}. These visitors convert best and cost the least to acquire.
# ==============================================================================

# ==============================================================================
# ## 2.5 How does website traffic differ between mobile and desktop users?
# ==============================================================================

# Aggregate sessions, orders and revenue by device type
device = sessions.groupby("device_type").agg(sessions=("website_session_id", "count"), orders=("converted", "sum"), revenue=("revenue_usd", "sum")).reset_index()
# Share of sessions by device
device["share_of_sessions_pct"] = device["sessions"] / device["sessions"].sum() * 100
# Share of orders by device
device["share_of_orders_pct"] = device["orders"] / device["orders"].sum() * 100
# Share of revenue by device
device["share_of_revenue_pct"] = device["revenue"] / device["revenue"].sum() * 100
# Conversion rate by device
device["conversion_rate_pct"] = device["orders"] / device["sessions"] * 100
# Revenue per session by device
device["revenue_per_session"] = device["revenue"] / device["sessions"]
# Average order value by device
device["aov"] = device["revenue"] / device["orders"]
# Display the device comparison
display(device)

# Conversion rate by quarter and device, in percent
dev_q = sessions.groupby(["quarter", "device_type"])["converted"].mean().unstack() * 100
# Share of mobile sessions by quarter, in percent
mobile_share_q = sessions.groupby("quarter")["device_type"].apply(lambda s: (s == "Mobile").mean() * 100)
# Device mix inside each channel (row percentages)
dev_channel = pd.crosstab(sessions["channel"], sessions["device_type"], normalize="index") * 100
# Display the device mix by channel
display(dev_channel)
# Conversion rate by year and device, in percent
dev_year = sessions.groupby(["year", "device_type"])["converted"].mean().unstack() * 100
# Display the yearly device conversion rates
display(dev_year)

# Create two charts side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Prepare data for the grouped bars: share of sessions, orders and revenue per device
share_cols = ["share_of_sessions_pct", "share_of_orders_pct", "share_of_revenue_pct"]
# Draw grouped bars for each device
device.set_index("device_type")[share_cols].plot(kind="bar", ax=axes[0], color=[PALETTE[0], PALETTE[2], PALETTE[4]], width=0.7)
# Write the percentage on top of each bar
for container in axes[0].containers:
    # Add rounded labels to every bar in the group
    axes[0].bar_label(container, fmt="%.0f%%", fontsize=9, padding=2)
# Add title and axis labels to the left chart
axes[0].set(title="Mobile brings traffic but far fewer orders", xlabel="", ylabel="Share (%)")
# Keep the device names horizontal
axes[0].tick_params(axis="x", rotation=0)
# Replace the long column names with readable legend labels
axes[0].legend(["% of sessions", "% of orders", "% of revenue"], frameon=False)
# Right chart: conversion rate over time for each device
dev_q.plot(ax=axes[1], marker="o", color=[PALETTE[0], PALETTE[3]], lw=2)
# Add title and axis labels to the right chart
axes[1].set(title="Conversion rate by quarter: desktop vs mobile", xlabel="", ylabel="Conversion rate (%)")
# Rotate the quarter labels
axes[1].tick_params(axis="x", rotation=60)
# Save and show the figure
save_fig("fig06_mobile_vs_desktop")

# Index the device table for lookup
dv = device.set_index("device_type")
# Store the device numbers used in the narrative
M.update(mobile_share=dv.loc["Mobile", "share_of_sessions_pct"] / 100, mobile_order_share=dv.loc["Mobile", "share_of_orders_pct"] / 100,
         mobile_cvr=dv.loc["Mobile", "conversion_rate_pct"], desktop_cvr=dv.loc["Desktop", "conversion_rate_pct"],
         cvr_ratio=dv.loc["Desktop", "conversion_rate_pct"] / dv.loc["Mobile", "conversion_rate_pct"],
         mobile_rps=dv.loc["Mobile", "revenue_per_session"], desktop_rps=dv.loc["Desktop", "revenue_per_session"],
         mobile_aov=dv.loc["Mobile", "aov"], desktop_aov=dv.loc["Desktop", "aov"],
         mobile_cvr_2012=dev_year.loc[2012, "Mobile"], mobile_cvr_2013=dev_year.loc[2013, "Mobile"], mobile_cvr_2014=dev_year.loc[2014, "Mobile"], mobile_cvr_2015=dev_year.loc[2015, "Mobile"],
         desktop_cvr_2012=dev_year.loc[2012, "Desktop"], desktop_cvr_2013=dev_year.loc[2013, "Desktop"], desktop_cvr_2014=dev_year.loc[2014, "Desktop"], desktop_cvr_2015=dev_year.loc[2015, "Desktop"])

# ==============================================================================
# Findings - mobile vs desktop
# - Mobile accounts for {...} of all sessions but only {...} of orders and revenue.
# - Desktop sessions convert at {...}% versus {...}% on mobile, a {...}x gap. Revenue per session is {...} on desktop against {...} on mobile, while order values are similar ({...} vs {...}), so the problem is *getting* mobile visitors to buy, not how much they spend.
# - The gap is persistent. Mobile conversion more than doubled from {...}% (2012) to {...}% (2013) but then plateaued ({...}% in 2014, {...}% in early 2015), while desktop kept climbing ({...}% in 2013, {...}% in 2014, {...}% in early 2015). Mobile has therefore been left behind by whatever is lifting desktop results.
# ==============================================================================

# ==============================================================================
# ## 2.6 What proportion of website sessions result in orders?
# ==============================================================================

# Overall session-to-order conversion rate
overall_cvr = sessions["converted"].mean()
# Print the overall conversion rate with the underlying counts
print(f"Overall: {sessions['converted'].sum():,} orders from {len(sessions):,} sessions = {pct(overall_cvr, 2)} conversion rate")
# Conversion rate by quarter for the trend chart
cvr_q = sessions.groupby("quarter")["converted"].mean() * 100
# Store the overall figure
M.update(overall_cvr=overall_cvr, cvr_q1_2013=cvr_q["2013Q1"], cvr_q4_2014=cvr_q["2014Q4"])
# Display the yearly conversion table created earlier
display(yearly[["sessions", "orders", "conversion_rate_pct"]])

# ==============================================================================
# ### Conversion funnel (page level)
# We now follow visitors page by page. This part uses only the {...} of sessions whose pageviews are fully present (see Section 1.3).
# ==============================================================================

# Keep only the sessions whose pageviews are complete (before the truncation point)
pv = pageviews[pageviews["website_session_id"] < PV_MAX_SESSION].sort_values("website_pageview_id").copy()
# Map every URL to a stage of the purchase funnel
STEP_MAP = {
    "/home": "Landing page", "/lander-1": "Landing page", "/lander-2": "Landing page", "/lander-3": "Landing page",
    "/lander-4": "Landing page", "/lander-5": "Landing page", "/products": "Products page",
    "/the-original-mr-fuzzy": "Product detail page", "/the-forever-love-bear": "Product detail page",
    "/the-birthday-sugar-panda": "Product detail page", "/the-hudson-river-mini-bear": "Product detail page",
    "/cart": "Cart", "/shipping": "Shipping", "/billing": "Billing", "/billing-2": "Billing",
    "/thank-you-for-your-order": "Order placed",
}
# Apply the mapping to every pageview
pv["step"] = pv["pageview_url"].map(STEP_MAP)
# Check that every URL was mapped to a funnel step
print("Unmapped pageviews:", int(pv["step"].isna().sum()))
# Order of the funnel stages from first to last
FUNNEL_ORDER = ["Landing page", "Products page", "Product detail page", "Cart", "Shipping", "Billing", "Order placed"]
# Count the distinct sessions that reached each stage
reach = pv.drop_duplicates(["website_session_id", "step"]).groupby("step")["website_session_id"].nunique().reindex(FUNNEL_ORDER)
# Build the funnel table
funnel = pd.DataFrame({"sessions": reach})
# Percentage of the previous stage that continued to this stage
funnel["pct_of_previous_step"] = funnel["sessions"] / funnel["sessions"].shift() * 100
# Percentage of all landing sessions that reached this stage
funnel["pct_of_all_sessions"] = funnel["sessions"] / funnel["sessions"].iloc[0] * 100
# Percentage lost between the previous stage and this one
funnel["drop_off_pct"] = 100 - funnel["pct_of_previous_step"]
# Display the funnel table
display(funnel)

# Create two charts side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.3, 1]})
# Left chart: horizontal funnel bars (first stage on top)
axes[0].barh(funnel.index[::-1], funnel["sessions"][::-1] / 1000, color=PALETTE[1])
# Annotate every bar with its count and the share retained from the previous step
for i, (name, r) in enumerate(funnel.iloc[::-1].iterrows()):
    # Build the label text (the first stage has no previous step)
    label = f"{r['sessions']/1000:,.0f}K" + ("" if np.isnan(r["pct_of_previous_step"]) else f"  ({r['pct_of_previous_step']:.0f}% of previous)")
    # Place the label to the right of the bar
    axes[0].text(r["sessions"] / 1000 + 4, i, label, va="center", fontsize=9)
# Add title and axis labels to the left chart
axes[0].set(title="Purchase funnel: where visitors drop out", xlabel="Sessions (thousands)", xlim=(0, funnel["sessions"].max() / 1000 * 1.4))
# Right chart: conversion rate by quarter
axes[1].plot(cvr_q.index, cvr_q.values, marker="o", color=PALETTE[0], lw=2)
# Add a dashed line for the overall average conversion rate
axes[1].axhline(overall_cvr * 100, ls="--", color="grey", label=f"Overall average {overall_cvr*100:.1f}%")
# Add title and axis labels to the right chart
axes[1].set(title="Session-to-order conversion rate by quarter", xlabel="", ylabel="Conversion rate (%)")
# Rotate the quarter labels
axes[1].tick_params(axis="x", rotation=60)
# Show the legend
axes[1].legend(frameon=False)
# Save and show the figure
save_fig("fig07_conversion_funnel_and_trend")

# First page of every session = its landing page
first_pv = pv.drop_duplicates("website_session_id")[["website_session_id", "pageview_url", "created_at"]].rename(columns={"pageview_url": "landing_page", "created_at": "landing_time"})
# Number of pages viewed in every session
pages_per_session = pv.groupby("website_session_id").size().rename("pages_viewed")
# Combine landing page, pages viewed, device, channel and the conversion flag
landing = first_pv.merge(pages_per_session, on="website_session_id").merge(sessions[["website_session_id", "converted", "device_type", "channel"]], on="website_session_id")
# A session "bounced" if it viewed only one page
landing["bounced"] = (landing["pages_viewed"] == 1).astype(int)
# Summarise performance per landing page over ALL traffic
landing_all = landing.groupby("landing_page").agg(sessions=("website_session_id", "count"), mobile_share_pct=("device_type", lambda s: (s == "Mobile").mean() * 100),
    bounce_rate_pct=("bounced", lambda s: s.mean() * 100), conversion_rate_pct=("converted", lambda s: s.mean() * 100), first_used=("landing_time", "min"), last_used=("landing_time", "max")).reset_index()
# Keep landing pages with a meaningful number of sessions and order them by first use
landing_all = landing_all[landing_all["sessions"] >= 500].sort_values("first_used")
# Display the overall landing page table (note the very different mobile shares)
display(landing_all)

# ==============================================================================
# Caution: the landing pages were not shown to comparable visitors. For example /lander-3 was served only to mobile visitors and /lander-4 and /lander-5 only to desktop visitors, so comparing their raw conversion rates would mainly compare *devices*. We therefore compare pages within the same device, using Google non-brand search traffic only (the audience that all lander pages share).
# ==============================================================================

# Keep only Google non-brand sessions, the audience shared by all lander pages, and the main landing pages
lp = landing[(landing["channel"] == "Google nonbrand") & (landing["landing_page"].isin(landing_all["landing_page"]))]
# Summarise by landing page and device
lp_tbl = lp.groupby(["landing_page", "device_type"]).agg(sessions=("website_session_id", "count"), bounce_rate_pct=("bounced", lambda s: s.mean() * 100),
    conversion_rate_pct=("converted", lambda s: s.mean() * 100), first_used=("landing_time", "min"), last_used=("landing_time", "max")).reset_index()
# Ignore combinations with very little traffic
lp_tbl = lp_tbl[lp_tbl["sessions"] >= 500].sort_values(["device_type", "first_used"])
# Display the like-for-like landing page table
display(lp_tbl)

# Create two charts side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
# Left chart: bounce rate by landing page, split by device
sns.barplot(data=lp_tbl, x="landing_page", y="bounce_rate_pct", hue="device_type", palette=[PALETTE[0], PALETTE[3]], ax=axes[0])
# Add title and axis labels to the left chart
axes[0].set(title="Bounce rate by landing page and device (Google non-brand)", xlabel="", ylabel="Bounce rate (%)")
# Right chart: conversion rate by landing page, split by device
sns.barplot(data=lp_tbl, x="landing_page", y="conversion_rate_pct", hue="device_type", palette=[PALETTE[0], PALETTE[3]], ax=axes[1])
# Add title and axis labels to the right chart
axes[1].set(title="Conversion rate by landing page and device (Google non-brand)", xlabel="", ylabel="Conversion rate (%)")
# Write the values above the bars of both charts
for ax in axes:
    # Add numeric labels to every bar group
    for container in ax.containers:
        # Label each bar with one decimal
        ax.bar_label(container, fmt="%.1f", padding=2, fontsize=8)
    # Place the legend outside the way of the bars
    ax.legend(title="Device", frameon=False)
# Save and show the figure
save_fig("fig08_landing_pages_by_device")

# Fair desktop test: compare /lander-2 and /lander-5 only while BOTH were live at the same time (Google non-brand, desktop)
d = landing[(landing["channel"] == "Google nonbrand") & (landing["device_type"] == "Desktop")]
# The overlap starts when /lander-5 first appeared
ov_start = d.loc[d["landing_page"] == "/lander-5", "landing_time"].min()
# The overlap ends when /lander-2 was last shown to desktop visitors
ov_end = d.loc[d["landing_page"] == "/lander-2", "landing_time"].max()
# Keep the two pages inside the overlap window
ov = d[d["landing_page"].isin(["/lander-2", "/lander-5"]) & d["landing_time"].between(ov_start, ov_end)]
# Summarise sessions, orders, conversion and bounce for each page
lander_ab = ov.groupby("landing_page").agg(sessions=("website_session_id", "count"), orders=("converted", "sum"), bounced=("bounced", "sum")).reset_index()
# Conversion rate and bounce rate of each page
lander_ab["conversion_rate_pct"] = lander_ab["orders"] / lander_ab["sessions"] * 100
# Bounce rate of each page
lander_ab["bounce_rate_pct"] = lander_ab["bounced"] / lander_ab["sessions"] * 100
# Print the overlap period and show the result
print(f"Overlap period: {ov_start.date()} to {ov_end.date()}")
# Display the like-for-like landing page A/B result
display(lander_ab)

# Funnel by device: sessions reaching each stage, split by mobile and desktop
dev_map = sessions.set_index("website_session_id")["device_type"]
# Attach the device to every funnel pageview (one row per session and step)
steps = pv.drop_duplicates(["website_session_id", "step"])[["website_session_id", "step"]].copy()
# Look up the device of each session
steps["device_type"] = steps["website_session_id"].map(dev_map)
# Count sessions per step and device and put the steps in funnel order
fd = steps.groupby(["step", "device_type"])["website_session_id"].nunique().unstack().reindex(FUNNEL_ORDER)
# Convert counts to "% of previous step" so that the two devices can be compared stage by stage
fd_rate = fd / fd.shift() * 100
# Rename the columns for clarity
fd_rate.columns = [f"{c} (% of previous step)" for c in fd_rate.columns]
# Display the stage-by-stage continuation rates by device
display(fd_rate.iloc[1:])

# Look at the two billing page versions: for every session keep the FIRST billing page it saw
bill_first = pv[pv["pageview_url"].isin(["/billing", "/billing-2"])].drop_duplicates("website_session_id")[["website_session_id", "pageview_url", "created_at"]]
# A fair A/B comparison only includes the period when BOTH versions were live at the same time
window_start = bill_first.loc[bill_first["pageview_url"] == "/billing-2", "created_at"].min()
# The end of the overlap is the last time the original billing page was shown
window_end = bill_first.loc[bill_first["pageview_url"] == "/billing", "created_at"].max()
# Keep only sessions in the overlap period
bill_win = bill_first[(bill_first["created_at"] >= window_start) & (bill_first["created_at"] <= window_end)].copy()
# Sessions that completed the order (reached the thank-you page)
placed_sessions = set(pv.loc[pv["step"] == "Order placed", "website_session_id"])
# Flag whether each billing session ended in an order
bill_win["ordered"] = bill_win["website_session_id"].isin(placed_sessions).astype(int)
# Summarise the two versions
bill_tbl = bill_win.groupby("pageview_url").agg(sessions=("website_session_id", "count"), orders=("ordered", "sum")).reset_index()
# Billing-to-order rate of each version
bill_tbl["billing_to_order_pct"] = bill_tbl["orders"] / bill_tbl["sessions"] * 100
# Print the overlap period used
print(f"Overlap period: {window_start.date()} to {window_end.date()}")
# Display the billing A/B table
display(bill_tbl)

# Store the funnel numbers used in the narrative
fn = funnel["pct_of_previous_step"]
# Pick the stage with the largest drop-off after the landing page
worst_step = funnel["drop_off_pct"].iloc[1:].idxmax()
# Index the billing table
bt = bill_tbl.set_index("pageview_url")
# Save the numbers
M.update(fn_products=fn["Products page"], fn_detail=fn["Product detail page"], fn_cart=fn["Cart"], fn_ship=fn["Shipping"], fn_bill=fn["Billing"], fn_order=fn["Order placed"],
         worst_step=worst_step, worst_drop=funnel.loc[worst_step, "drop_off_pct"], bounce_overall=landing["bounced"].mean(),
         bill1=bt.loc["/billing", "billing_to_order_pct"], bill2=bt.loc["/billing-2", "billing_to_order_pct"],
         bill_window=f"{window_start.strftime('%d %b %Y')} to {window_end.strftime('%d %b %Y')}", bill_n=int(bt["sessions"].sum()))
# Index the like-for-like landing page table by page and device
lpi = lp_tbl.set_index(["landing_page", "device_type"])
# Gap in continuation rate between desktop and mobile at every funnel stage (percentage points)
gap = (fd_rate.iloc[1:, 0] - fd_rate.iloc[1:, 1]).sort_values(ascending=False)
# Store the landing-page and device-funnel numbers
M.update(l5_desk_cvr=lpi.loc[("/lander-5", "Desktop"), "conversion_rate_pct"], l5_desk_bounce=lpi.loc[("/lander-5", "Desktop"), "bounce_rate_pct"],
         l5_first=lpi.loc[("/lander-5", "Desktop"), "first_used"], home_desk_cvr=lpi.loc[("/home", "Desktop"), "conversion_rate_pct"],
         home_mob_cvr=lpi.loc[("/home", "Mobile"), "conversion_rate_pct"], l3_mob_cvr=lpi.loc[("/lander-3", "Mobile"), "conversion_rate_pct"],
         l3_mob_bounce=lpi.loc[("/lander-3", "Mobile"), "bounce_rate_pct"], home_mob_bounce=lpi.loc[("/home", "Mobile"), "bounce_rate_pct"],
         la2_cvr=lander_ab.set_index("landing_page").loc["/lander-2", "conversion_rate_pct"], la5_cvr=lander_ab.set_index("landing_page").loc["/lander-5", "conversion_rate_pct"],
         la2_bounce=lander_ab.set_index("landing_page").loc["/lander-2", "bounce_rate_pct"], la5_bounce=lander_ab.set_index("landing_page").loc["/lander-5", "bounce_rate_pct"],
         la_window=f"{ov_start.strftime('%d %b %Y')} to {ov_end.strftime('%d %b %Y')}", la_n=int(lander_ab["sessions"].sum()),
         mob_best_cvr=lpi.loc[("/lander-3", "Mobile"), "conversion_rate_pct"], desk_best_cvr=lpi.loc[("/lander-5", "Desktop"), "conversion_rate_pct"],
         gap_step=gap.index[0], gap_desk=fd_rate.loc[gap.index[0]].iloc[0], gap_mob=fd_rate.loc[gap.index[0]].iloc[1],
         gap2_step=gap.index[1], gap2_desk=fd_rate.loc[gap.index[1]].iloc[0], gap2_mob=fd_rate.loc[gap.index[1]].iloc[1])

# ==============================================================================
# Findings - sessions to orders
# - {...} of all sessions end in an order ({...} orders from {...} sessions). The rate climbed from {...}% in 2012 to {...}% in 2014 and {...}% in early 2015.
# - Funnel (each figure is the share of the previous stage): {...}% of landing sessions reach the products page, {...}% of those open a product, {...}% go on to the cart, {...}% continue to shipping, {...}% reach billing and {...}% complete the order. The two biggest leaks are product page to cart ({...}% leave) and landing page to products page ({...}% leave).
# - The biggest desktop-vs-mobile gap is at the step into "{...}": {...}% of desktop sessions continue to that stage versus {...}% of mobile sessions (next largest gap: "{...}", {...}% vs {...}%). Mobile visitors fall behind at *every* stage, not at a single broken step.
# - Bounce rate is {...} overall: nearly half of all visitors leave after one page, so landing-page quality matters.
# - Landing pages: pages were served to different devices and periods, so we compare like with like. On desktop, while /lander-2 and /lander-5 ran side by side ({...}, {...} Google non-brand sessions), /lander-5 converted {...}% vs {...}% and bounced {...}% vs {...}% (significance tested in Task 3). On mobile, even the best (latest) page, /lander-3, converts only {...}% against {...}% for the best desktop page, so landing-page design alone has not closed the mobile gap.
# ==============================================================================

# ==============================================================================
# ## 2.7 Which products have the highest refund activity?
# ==============================================================================

# Display refund activity by product (refund count, rate and dollar value)
refund_tbl = prod[["product_name", "short_name", "units", "refunded_units", "refund_rate_pct", "refunds"]].copy()
# Share of all refund dollars that each product causes
refund_tbl["share_of_refund_dollars_pct"] = refund_tbl["refunds"] / refund_tbl["refunds"].sum() * 100
# Sort by refund rate, highest first
refund_tbl = refund_tbl.sort_values("refund_rate_pct", ascending=False)
# Display the refund table
display(refund_tbl)

# Days between order and refund for every refunded item
lag = refunds.merge(items[["order_item_id", "created_at"]], on="order_item_id", suffixes=("_refund", "_order"))
# Convert the time difference into days
lag["days_to_refund"] = (lag["created_at_refund"] - lag["created_at_order"]).dt.days
# Share of refunds that equal the full item price (full refunds)
full_refund = refunds.merge(order_items[["order_item_id", "price_usd"]], on="order_item_id")
# Compute the full-refund share
full_share = np.isclose(full_refund["refund_amount_usd"], full_refund["price_usd"]).mean()
# Overall refund rate and value as a share of revenue
overall_refund_rate = items["is_refunded"].mean()
# Refund dollars as a share of gross revenue
refund_pct_rev = refunds["refund_amount_usd"].sum() / orders["price_usd"].sum()
# Print the refund statistics
print(f"Refunded items: {int(items['is_refunded'].sum()):,} of {len(items):,} ({pct(overall_refund_rate)}) | refund value {money(refunds['refund_amount_usd'].sum())} = {pct(refund_pct_rev)} of gross revenue")
# Print the timing and size of refunds
print(f"Refund timing: median {lag['days_to_refund'].median():.0f} days after order (range {lag['days_to_refund'].min()}-{lag['days_to_refund'].max()}) | full refunds: {pct(full_share, 0)}")
# Refund rate by order month and product (months with at least 100 items sold)
rate_m = items.groupby(["month", "short_name"]).agg(items_sold=("order_item_id", "count"), refund_rate=("is_refunded", "mean")).reset_index()
# Keep months with enough volume and complete months only
rate_m = rate_m[(rate_m["items_sold"] >= 100) & (rate_m["month"] <= LAST_FULL_MONTH)]
# Store the results
M.update(overall_refund_rate=overall_refund_rate, refund_pct_rev=refund_pct_rev, refund_lag_median=lag["days_to_refund"].median(),
         full_refund_share=full_share, refund_total_items=int(items["is_refunded"].sum()))

# Create two charts side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Left chart: refund rate by product, highest first
sns.barplot(data=refund_tbl, x="short_name", y="refund_rate_pct", color=PALETTE[3], ax=axes[0])
# Draw the overall refund rate as a dashed reference line
axes[0].axhline(overall_refund_rate * 100, ls="--", color="grey", label=f"All products {overall_refund_rate*100:.1f}%")
# Write the values above the bars
axes[0].bar_label(axes[0].containers[0], fmt="%.1f%%", padding=2)
# Add title and axis labels to the left chart
axes[0].set(title="Refund rate by product", xlabel="", ylabel="Refunded units (% of units sold)")
# Show the legend
axes[0].legend(frameon=False)
# Right chart: monthly refund rate per product, in percent
sns.lineplot(data=rate_m.assign(refund_rate=rate_m["refund_rate"] * 100), x="month", y="refund_rate", hue="short_name", palette=PALETTE[:4], marker="o", ms=4, ax=axes[1])
# Add title and axis labels to the right chart
axes[1].set(title="Monthly refund rate by order month", xlabel="", ylabel="Refund rate (%)")
# Show a clean legend
axes[1].legend(title="Product", frameon=False)
# Save and show the figure
save_fig("fig09_refunds_by_product")

# Monthly refund history of the best seller
fuzzy_m = rate_m[rate_m["short_name"] == "Mr. Fuzzy"].set_index("month")
# The two months with the highest refund rate (candidates for a refund "episode")
spike = fuzzy_m.nlargest(2, "refund_rate")
# Start and end of the spike
spike_start, spike_end = spike.index.min(), spike.index.max()
# Baseline = average refund rate in the surrounding months of the same period (Jan 2014 - Feb 2015, excluding the spike)
base = fuzzy_m.loc["2014-01-01":"2015-02-01"].drop(spike.index)
# Baseline refund rate weighted by items sold
base_rate = (base["refund_rate"] * base["items_sold"]).sum() / base["items_sold"].sum()
# Actual refunded items during the spike
actual = (spike["refund_rate"] * spike["items_sold"]).sum()
# Refunded items that would be expected at the baseline rate
expected = base_rate * spike["items_sold"].sum()
# Excess refunds and their dollar value (full refunds at the item price of 49.99)
excess_items = actual - expected
# Dollar value of the excess refunds
excess_usd = excess_items * items.loc[items["short_name"] == "Mr. Fuzzy", "price_usd"].median()
# Refund rate of Mr. Fuzzy by year (all orders)
refund_year = items.groupby(["year", "short_name"])["is_refunded"].mean().unstack() * 100
# Display refund rate by year and product
display(refund_year)
# Print the spike analysis
print(f"Spike months: {spike_start:%b %Y} - {spike_end:%b %Y} | rates: {', '.join(f'{v*100:.1f}%' for v in spike['refund_rate'])} | baseline {base_rate*100:.1f}%")
# Print the estimated excess
print(f"Estimated excess refunds: {excess_items:,.0f} items = {money(excess_usd)}")
# Index the refund table for lookup
rt = refund_tbl.set_index("short_name")
# Save results
M.update(spike_months=f"{spike_start:%B}-{spike_end:%B %Y}", spike_rate=spike["refund_rate"].mean(), base_rate=base_rate, excess_items=excess_items, excess_usd=excess_usd,
         fuzzy_refund_2012=refund_year.loc[2012, "Mr. Fuzzy"], fuzzy_refund_2014=refund_year.loc[2014, "Mr. Fuzzy"],
         panda_refund=rt.loc["Sugar Panda", "refund_rate_pct"], fuzzy_refund=rt.loc["Mr. Fuzzy", "refund_rate_pct"], love_refund=rt.loc["Love Bear", "refund_rate_pct"],
         mini_refund=rt.loc["Mini Bear", "refund_rate_pct"], fuzzy_refund_share=rt.loc["Mr. Fuzzy", "share_of_refund_dollars_pct"] / 100,
         refund_top_name=refund_tbl.iloc[0]["short_name"])

# ==============================================================================
# Findings - refunds
# - Overall, {...} of units sold are refunded ({...} items), costing {...} or {...} of gross revenue. Refunds are quick (median {...} days after the order) and {...} are full refunds.
# - Highest refund rate: {...} ({...}%), followed by Mr. Fuzzy ({...}%), Love Bear ({...}%) and Mini Bear ({...}%).
# - Highest refund dollars: Mr. Fuzzy ({...} of all refund dollars) simply because it sells the most units.
# - A refund episode hit Mr. Fuzzy in {...}: the refund rate jumped to about {...} versus a normal {...}, then fell straight back. That is roughly {...} extra refunds worth about {...}. A sudden, temporary spike like this points to a product-quality or supplier problem (for example one defective batch) rather than a permanent customer preference, and the fast recovery suggests it was fixed.
# - Long-run improvement: the Mr. Fuzzy refund rate fell from {...}% in 2012 to {...}% in 2014 (including the spike), while Sugar Panda stays the structurally highest-refund product.
# ==============================================================================

# ==============================================================================
# ## 2.8 Are there noticeable differences between new and returning website sessions?
# ==============================================================================

# Aggregate sessions, orders and revenue by visitor type (new vs returning)
visitor = sessions.groupby("visitor_type").agg(sessions=("website_session_id", "count"), orders=("converted", "sum"), revenue=("revenue_usd", "sum")).reset_index()
# Share of sessions by visitor type
visitor["share_of_sessions_pct"] = visitor["sessions"] / visitor["sessions"].sum() * 100
# Conversion rate by visitor type
visitor["conversion_rate_pct"] = visitor["orders"] / visitor["sessions"] * 100
# Revenue per session by visitor type
visitor["revenue_per_session"] = visitor["revenue"] / visitor["sessions"]
# Average order value by visitor type
visitor["aov"] = visitor["revenue"] / visitor["orders"]
# Display the new vs returning comparison
display(visitor)

# Channel mix of new versus returning sessions (column percentages)
mix_channel = pd.crosstab(sessions["channel"], sessions["visitor_type"], normalize="columns") * 100
# Device mix of new versus returning sessions (column percentages)
mix_device = pd.crosstab(sessions["device_type"], sessions["visitor_type"], normalize="columns") * 100
# Share of returning sessions by quarter
ret_q = sessions.groupby("quarter")["is_repeat_session"].mean() * 100
# Display the channel mix
display(mix_channel)
# Display the device mix
display(mix_device)

# New and returning visitors do not arrive through the same channels, so compare them within the same (brand-aware) channels
shared = sessions[sessions["channel"].isin(["Direct", "Organic search", "Paid brand search"])]
# Summarise sessions, orders and conversion rate by channel and visitor type
shared_tbl = shared.groupby(["channel", "visitor_type"]).agg(sessions=("website_session_id", "count"), orders=("converted", "sum"), conversion_rate_pct=("converted", lambda s: s.mean() * 100)).reset_index()
# Display the like-for-like comparison inside each channel
display(shared_tbl)
# Pool the three channels by visitor type
shared_vt = shared.groupby("visitor_type").agg(sessions=("website_session_id", "count"), orders=("converted", "sum"))
# Conversion rate of new and returning visitors within the same channels
shared_vt["conversion_rate_pct"] = shared_vt["orders"] / shared_vt["sessions"] * 100
# Display the pooled like-for-like comparison
display(shared_vt)

# Create two charts side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
# Left chart: conversion rate and revenue per session for new vs returning visitors
vt = visitor.set_index("visitor_type")
# Draw conversion-rate bars
axes[0].bar(vt.index, vt["conversion_rate_pct"], color=[PALETTE[1], PALETTE[2]], width=0.5)
# Write the values on top of the bars
axes[0].bar_label(axes[0].containers[0], fmt="%.1f%%", padding=2)
# Add title and axis labels to the left chart
axes[0].set(title="Returning visitors convert better", ylabel="Conversion rate (%)", xlabel="", ylim=(0, vt["conversion_rate_pct"].max() * 1.25))
# Right chart: share of returning sessions over time
axes[1].plot(ret_q.index, ret_q.values, marker="o", color=PALETTE[2], lw=2)
# Add title and axis labels to the right chart
axes[1].set(title="Returning visitors as % of sessions", xlabel="", ylabel="% of sessions")
# Rotate the quarter labels
axes[1].tick_params(axis="x", rotation=60)
# Save and show the figure
save_fig("fig10_new_vs_returning")

# Index the visitor table for lookup
vv = visitor.set_index("visitor_type")
# Store the numbers used in the narrative
M.update(new_cvr=vv.loc["New", "conversion_rate_pct"], ret_cvr=vv.loc["Returning", "conversion_rate_pct"], ret_share=vv.loc["Returning", "share_of_sessions_pct"] / 100,
         new_rps=vv.loc["New", "revenue_per_session"], ret_rps=vv.loc["Returning", "revenue_per_session"], new_aov=vv.loc["New", "aov"], ret_aov=vv.loc["Returning", "aov"],
         ret_q_2012=ret_q["2012Q2"], ret_q_2014q4=ret_q["2014Q4"], ret_mobile=mix_device.loc["Mobile", "Returning"] / 100, new_mobile=mix_device.loc["Mobile", "New"] / 100,
         ret_direct=mix_channel.loc["Direct", "Returning"] / 100, ret_organic=mix_channel.loc["Organic search", "Returning"] / 100,
         ret_brand=mix_channel.loc["Paid brand search", "Returning"] / 100, shared_new_cvr=shared_vt.loc["New", "conversion_rate_pct"], shared_ret_cvr=shared_vt.loc["Returning", "conversion_rate_pct"])
# Write the verdict sentence depending on how large the like-for-like gap is
gap_pp = M["shared_ret_cvr"] - M["shared_new_cvr"]
# A gap of under 0.5 percentage points is treated as small
M["shared_verdict"] = ("most of the headline advantage of returning visitors comes from the channels they use, not from being returning visitors. The extra conversion attributable to returning status is small (" + f"{gap_pp:+.1f}" + " percentage points)." if abs(gap_pp) < 0.5
                       else "returning visitors keep a genuine edge of " + f"{gap_pp:+.1f}" + " percentage points even inside the same channels (tested in Task 3).")

# ==============================================================================
# Findings - new vs returning sessions
# - Returning visitors are {...} of sessions but convert better: {...}% vs {...}% for new visitors, and earn {...} vs {...} revenue per session. Order values are similar ({...} vs {...}), so the benefit comes from higher purchase likelihood.
# - The returning share is rising ({...}% of sessions in 2012Q2 to {...}% in 2014Q4), a sign of growing brand awareness.
# - Returning sessions arrive only through brand-aware channels (direct {...}, organic {...}, paid brand {...}); none are attributed to non-brand search or social. They are also more mobile ({...} of returning vs {...} of new sessions).
# - Like-for-like check: inside those same three channels, new visitors convert at {...}% and returning visitors at {...}%. So {...}
# ==============================================================================

# ==============================================================================
# # Task 3: Statistical Analysis and Visualisation
# ## 3.1 Descriptive statistics
# ==============================================================================

# Choose the numeric order-level columns to describe
num_cols = ["price_usd", "items_purchased", "cogs_usd", "gross_profit_usd", "order_refund_usd"]
# Standard descriptive statistics (count, mean, std, quartiles, min, max)
desc = orders[num_cols].describe().T
# Add the median explicitly (it equals the 50% quartile) for clarity
desc["median"] = orders[num_cols].median()
# Add skewness: > 0 means a long tail of large values
desc["skewness"] = orders[num_cols].skew()
# Add the coefficient of variation (std / mean): a scale-free measure of spread
desc["coef_variation"] = desc["std"] / desc["mean"]
# Display the order-level descriptive statistics
display(desc)

# Build a daily table of sessions, orders and revenue
daily = pd.concat([sessions.groupby("date").size().rename("sessions"),
                   orders.groupby("date").agg(orders=("order_id", "count"), revenue=("price_usd", "sum"))], axis=1).fillna(0)
# Keep only complete days (the first and last day of the data are partial)
daily = daily.loc[FIRST_FULL_DAY:LAST_FULL_DAY].copy()
# Daily conversion rate in percent
daily["cvr_pct"] = daily["orders"] / daily["sessions"] * 100
# Calendar year of every day
daily["year"] = daily.index.year
# Descriptive statistics of the daily metrics for the two complete years
daily_desc = daily[daily["year"].isin([2013, 2014])].groupby("year")[["sessions", "orders", "revenue", "cvr_pct"]].agg(["mean", "median", "std", "min", "max"]).T
# Display the daily statistics by year
display(daily_desc)

# Descriptive statistics of pages viewed per session (sessions with complete pageview data)
pps = pages_per_session.describe()
# Share of sessions that viewed only one page
single_page = (pages_per_session == 1).mean()
# Print the pages-per-session statistics
print(f"Pages per session: mean {pps['mean']:.2f}, median {pps['50%']:.0f}, max {pps['max']:.0f} | single-page sessions: {pct(single_page)}")
# Frequency table of the categorical variables (share of sessions)
for col in ["device_type", "visitor_type", "channel"]:
    # Show the percentage distribution of each categorical column
    print("\n" + (sessions[col].value_counts(normalize=True) * 100).round(1).to_string())
# Store the daily means for the report
M.update(daily_orders_2014=daily.loc[daily["year"] == 2014, "orders"].mean(), daily_rev_2014=daily.loc[daily["year"] == 2014, "revenue"].mean(),
         daily_sessions_2014=daily.loc[daily["year"] == 2014, "sessions"].mean(), order_median=orders["price_usd"].median(), order_mean=orders["price_usd"].mean())

# ==============================================================================
# Interpretation
# - The typical order is {...} (median) and the mean is {...}. The mean is higher than the median because of the two-item orders: the order-value distribution is *right-skewed* with only two distinct sizes (one item, two items).
# - In 2014 the shop received about {...} sessions, {...} orders and {...} revenue per day on average.
# - Daily revenue has a high coefficient of variation because of the strong upward trend and the weekday/holiday seasonality analysed below.
# ==============================================================================

# ==============================================================================
# ## 3.2 Relationship between website traffic and orders
# ==============================================================================

# Pearson correlation (linear relationship) between daily sessions and daily orders
pear = stats.pearsonr(daily["sessions"], daily["orders"])
# Spearman correlation (rank-based, robust to outliers and non-linearity)
spear = stats.spearmanr(daily["sessions"], daily["orders"])
# Simple linear regression: orders = intercept + slope * sessions
reg = stats.linregress(daily["sessions"], daily["orders"])
# Print the overall correlation results
print(f"Pearson r = {pear.statistic:.3f} (p {'< 0.001' if pear.pvalue < 0.001 else '= ' + format(pear.pvalue, '.3f')}) | Spearman rho = {spear.statistic:.3f} | R-squared = {reg.rvalue**2:.3f}")
# Print the regression slope converted to "orders per 100 additional sessions"
print(f"Regression slope: {reg.slope * 100:.2f} extra orders for every 100 extra sessions")
# Both series trend upwards over time, so also test the relationship WITHIN each year
within = {int(y): float(g["sessions"].corr(g["orders"])) for y, g in daily.groupby("year")}
# Print the within-year correlations
print("Within-year Pearson r:", {k: round(v, 3) for k, v in within.items()})
# And test the day-to-day CHANGES, which removes the common time trend
diff_corr = daily[["sessions", "orders"]].diff().dropna().corr().iloc[0, 1]
# Print the correlation of daily changes
print(f"Correlation of day-to-day changes: {diff_corr:.3f}")
# Weekly totals smooth out weekday effects
weekly = daily[["sessions", "orders"]].resample("W").sum()
# Print the weekly correlation
print(f"Weekly totals Pearson r: {weekly['sessions'].corr(weekly['orders']):.3f}")
# Store the results
M.update(r_daily=pear.statistic, r_p=pear.pvalue, r2_daily=reg.rvalue ** 2, slope_per100=reg.slope * 100, r_diff=diff_corr, r_week=weekly["sessions"].corr(weekly["orders"]),
         r_within_min=min(within.values()), r_within_max=max(within.values()))

# Create the scatter plot of daily sessions versus daily orders
fig, ax = plt.subplots(figsize=(9.5, 6))
# Draw one point per day, coloured by year to reveal the growth over time
sns.scatterplot(data=daily.reset_index(), x="sessions", y="orders", hue="year", palette="viridis", s=22, alpha=0.75, ax=ax)
# Create x values for the regression line
xs = np.linspace(daily["sessions"].min(), daily["sessions"].max(), 100)
# Draw the fitted regression line
ax.plot(xs, reg.intercept + reg.slope * xs, color=PALETTE[3], lw=2.5, label=f"Linear fit (r = {pear.statistic:.2f})")
# Add title and axis labels
ax.set(title="More traffic, more orders: daily sessions vs daily orders", xlabel="Sessions per day", ylabel="Orders per day")
# Show the legend
ax.legend(title="Year", frameon=False)
# Save and show the figure
save_fig("fig11_traffic_vs_orders")

# ==============================================================================
# Interpretation. Daily sessions and daily orders are very strongly correlated (Pearson r = {...}, R² = {...}, p < 0.001): every extra 100 sessions bring about {...} extra orders. Because both series trend upward, we double-checked: the relationship stays strong within each year (r between {...} and {...}), in day-to-day changes (r = {...}) and in weekly totals (r = {...}). So traffic genuinely drives orders; the points drifting upward by colour also show that each year converts more traffic into orders than the one before. (Correlation shows association, not proof of causation.)
# ==============================================================================

# ==============================================================================
# ## 3.3 Comparing sales across different periods
# ==============================================================================

# Like-for-like comparison: only 1 January to 18 March of each year (data ends on 19 March 2015)
def like_for_like(df):
    """Return True for rows between 1 Jan and 18 Mar (inclusive) of any year."""
    # Month of every row
    m = df["created_at"].dt.month
    # Day of the month of every row
    d = df["created_at"].dt.day
    # Months January and February, or March up to the 18th
    return (m < 3) | ((m == 3) & (d <= 18))

# Revenue and orders of the like-for-like window per year (2012 has no data before 19 March)
lfl_o = orders[like_for_like(orders) & (orders["year"] >= 2013)].groupby("year").agg(orders=("order_id", "count"), revenue=("price_usd", "sum"))
# Sessions of the like-for-like window per year
lfl_s = sessions[like_for_like(sessions) & (sessions["year"] >= 2013)].groupby("year").size().rename("sessions")
# Combine the tables
lfl = lfl_o.join(lfl_s)
# Conversion rate and growth rates versus the previous year
lfl["conversion_rate_pct"] = lfl["orders"] / lfl["sessions"] * 100
# Year-over-year revenue growth in percent
lfl["revenue_yoy_pct"] = lfl["revenue"].pct_change() * 100
# Year-over-year session growth in percent
lfl["sessions_yoy_pct"] = lfl["sessions"].pct_change() * 100
# Display the like-for-like (1 Jan - 18 Mar) comparison
display(lfl)
# Store the key result
M.update(lfl_15_vs_14=lfl.loc[2015, "revenue_yoy_pct"] / 100, lfl_14_vs_13=lfl.loc[2014, "revenue_yoy_pct"] / 100)

# Average DAILY revenue for each bimonthly period, so periods of different length are comparable
daily_rev = orders.groupby("date")["price_usd"].sum().reindex(pd.date_range(FIRST_FULL_DAY, LAST_FULL_DAY), fill_value=0)
# Helper that returns the daily revenue values for given months of a given year
def period(year, months):
    """Daily revenue values for the given months of a year."""
    # Select days of the requested year and months
    return daily_rev[(daily_rev.index.year == year) & (daily_rev.index.month.isin(months))]
# Collect the seasonal comparison for the full years 2013 and 2014 (and 2012 where available)
rows = []
# Loop over the years that contain the whole Jul-Dec period
for yr in [2012, 2013, 2014]:
    # Average daily revenue in July-August, September-October and November-December
    ja, so, nd = period(yr, [7, 8]).mean(), period(yr, [9, 10]).mean(), period(yr, [11, 12]).mean()
    # Mann-Whitney test: are Nov-Dec days different from Sep-Oct days?
    mw = stats.mannwhitneyu(period(yr, [11, 12]), period(yr, [9, 10]), alternative="two-sided")
    # Store the figures
    rows.append({"year": yr, "jul_aug_daily_rev": ja, "sep_oct_daily_rev": so, "nov_dec_daily_rev": nd,
                 "trend_growth_pct (SepOct vs JulAug)": (so / ja - 1) * 100, "holiday_uplift_pct (NovDec vs SepOct)": (nd / so - 1) * 100,
                 "excess_uplift_pp": ((nd / so) - (so / ja)) * 100, "mann_whitney_p": mw.pvalue})
# Convert to a DataFrame
season = pd.DataFrame(rows).set_index("year")
# Display the seasonality table
display(season)
# Store results (2013 and 2014 are the clean full years)
M.update(uplift_13=season.loc[2013, "holiday_uplift_pct (NovDec vs SepOct)"] / 100, uplift_14=season.loc[2014, "holiday_uplift_pct (NovDec vs SepOct)"] / 100,
         trend_14=season.loc[2014, "trend_growth_pct (SepOct vs JulAug)"] / 100, excess_14=season.loc[2014, "excess_uplift_pp"], excess_13=season.loc[2013, "excess_uplift_pp"],
         dec14_daily=period(2014, [12]).mean(), sepoct14_daily=season.loc[2014, "sep_oct_daily_rev"])

# Compare weekday and weekend revenue inside the single full year 2014 (so the long-term trend does not distort the test)
d14 = daily_rev[daily_rev.index.year == 2014]
# Daily revenue on weekdays (Monday-Friday)
wd = d14[d14.index.dayofweek < 5]
# Daily revenue on weekend days (Saturday-Sunday)
we = d14[d14.index.dayofweek >= 5]
# Non-parametric test, because daily revenue is not normally distributed
mw_week = stats.mannwhitneyu(wd, we, alternative="two-sided")
# Print the comparison
print(f"2014 average daily revenue: weekday {money(wd.mean())} vs weekend {money(we.mean())} (weekend = {we.mean() / wd.mean() * 100:.0f}% of a weekday) | Mann-Whitney p = {mw_week.pvalue:.1e}")
# Store the results
M.update(wd_rev=wd.mean(), we_rev=we.mean(), we_ratio=we.mean() / wd.mean(), mw_week_p=mw_week.pvalue)

# Create two charts side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Left chart: quarterly revenue for complete quarters with year-over-year growth labels
qc = qtr[qtr["complete"]].copy()
# Draw the bars
bars = axes[0].bar(qc["quarter"], qc["gross_revenue"] / 1000, color=PALETTE[1])
# Annotate each bar with its YoY growth (when available)
for bar, yoy in zip(bars, qc["yoy_growth_pct"]):
    # Skip the first four quarters that have no comparable previous-year quarter
    if not np.isnan(yoy):
        # Write the growth percentage above the bar
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4, f"+{yoy:.0f}%", ha="center", fontsize=8, color="dimgrey")
# Add title and labels to the left chart
axes[0].set(title="Quarterly revenue and year-over-year growth", xlabel="", ylabel="Gross revenue ($ thousands)")
# Rotate the quarter labels
axes[0].tick_params(axis="x", rotation=60)
# Right chart: average daily revenue by calendar month for 2013 and 2014 (shows the seasonal shape)
seas = orders[orders["year"].isin([2013, 2014])].groupby(["year", orders["created_at"].dt.month])["price_usd"].sum().unstack(0)
# Convert monthly totals into daily averages by dividing by days in the month
seas = seas.div(pd.Series([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], index=range(1, 13)), axis=0)
# Draw one line per year
seas.plot(ax=axes[1], marker="o", color=[PALETTE[1], PALETTE[3]], lw=2)
# Label the months on the x-axis
axes[1].set_xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
# Add title and labels to the right chart
axes[1].set(title="Average daily revenue by month: a clear Nov-Dec peak", xlabel="", ylabel="Revenue per day ($)")
# Show the legend
axes[1].legend(title="Year", frameon=False)
# Save and show the figure
save_fig("fig12_period_comparison")

# ==============================================================================
# Interpretation
# - Year over year: in the like-for-like window (1 Jan - 18 Mar) revenue grew {...} in 2014 and still {...} in 2015, so growth is sustained, not slowing down sharply.
# - Holiday seasonality: average daily revenue in November-December was {...} above September-October in 2014 ({...} in 2013). Because the business was also growing anyway ({...} from July-August to September-October in 2014), the *extra* holiday effect is roughly {...} percentage points in 2014. The difference is statistically significant (see test table).
# - Weekday vs weekend: in 2014 a weekend day brought only {...} of the revenue of a weekday ({...} vs {...}), clearly significant (Mann-Whitney p < 0.001).
# ==============================================================================

# ==============================================================================
# ## 3.4 Relationship between product sales and refunds
# ==============================================================================

# Contingency table: number of items refunded vs not refunded for every product
ct = pd.crosstab(items["short_name"], items["is_refunded"]).rename(columns={0: "not_refunded", 1: "refunded"})
# Chi-square test of independence: is the refund rate the same for all products?
chi = stats.chi2_contingency(ct)
# Cramer's V: effect size of the association between product and refund (0 = none, 1 = perfect)
cramers_v = np.sqrt(chi.statistic / (ct.values.sum() * (min(ct.shape) - 1)))
# Standardised residuals: positive = more refunds than expected, negative = fewer
resid = pd.DataFrame((ct.values - chi.expected_freq) / np.sqrt(chi.expected_freq), index=ct.index, columns=ct.columns)
# Print the test result
print(f"Chi-square = {chi.statistic:,.1f}, dof = {chi.dof}, p = {chi.pvalue:.1e}, Cramer's V = {cramers_v:.3f}")
# Display the counts and the residuals
display(ct)
# Display the standardised residuals of the refunded column
display(resid[["refunded"]].rename(columns={"refunded": "std_residual_refunded"}))

# Refund rate of primary items versus cross-sell (second) items, only after cross-selling started
post_items = items[items["created_at"] >= first_bundle]
# Summarise the two roles
role_tbl = post_items.groupby("item_role").agg(items=("order_item_id", "count"), refunded=("is_refunded", "sum"))
# Refund rate of each role
role_tbl["refund_rate_pct"] = role_tbl["refunded"] / role_tbl["items"] * 100
# Display the role comparison
display(role_tbl)
# Product-month table: units sold and units refunded per product and order month (complete months, products with at least 50 sales)
pm = items.groupby(["short_name", "month"]).agg(units=("order_item_id", "count"), refunded=("is_refunded", "sum")).reset_index()
# Keep complete months with enough volume
pm = pm[(pm["units"] >= 50) & (pm["month"] <= LAST_FULL_MONTH)].copy()
# Refund rate per product-month
pm["rate"] = pm["refunded"] / pm["units"]
# Correlation between units sold and refunded units: refunds scale with sales
r_units = stats.pearsonr(pm["units"], pm["refunded"])
# Correlation between units sold and the refund RATE: do bigger months have worse refund rates?
r_rate = stats.spearmanr(pm["units"], pm["rate"])
# Print both results
print(f"Units sold vs refunded units: Pearson r = {r_units.statistic:.2f} (p = {r_units.pvalue:.1e}, n = {len(pm)} product-months)")
# Print the second result
print(f"Units sold vs refund RATE: Spearman rho = {r_rate.statistic:.2f} (p = {r_rate.pvalue:.2f})")
# Store the results
M.update(chi_prod=chi.statistic, chi_prod_p=chi.pvalue, cramers_v=cramers_v, r_units=r_units.statistic, rho_rate=r_rate.statistic, rho_rate_p=r_rate.pvalue,
         primary_refund=role_tbl.loc["Primary item", "refund_rate_pct"], cross_refund=role_tbl.loc["Cross-sell item", "refund_rate_pct"])

# Create the scatter plot of units sold versus units refunded
fig, ax = plt.subplots(figsize=(9.5, 6))
# Draw one point per product-month, coloured by product
sns.scatterplot(data=pm, x="units", y="refunded", hue="short_name", palette=PALETTE[:4], s=60, alpha=0.85, ax=ax)
# Highlight the Aug-Sep 2014 Mr. Fuzzy refund spike
spk = pm[(pm["short_name"] == "Mr. Fuzzy") & (pm["month"].isin(spike.index))]
# Circle the two spike points
ax.scatter(spk["units"], spk["refunded"], s=260, facecolors="none", edgecolors="black", linewidths=1.6, label="Aug-Sep 2014 spike")
# Overall linear trend line (refunds scale with sales)
slope, intercept = np.polyfit(pm["units"], pm["refunded"], 1)
# Draw the trend line
xs = np.linspace(pm["units"].min(), pm["units"].max(), 50)
# Plot the line
ax.plot(xs, intercept + slope * xs, ls="--", color="grey", lw=1.5)
# Add title and axis labels
ax.set(title="Refunds rise with sales volume; the 2014 Mr. Fuzzy spike is the outlier", xlabel="Units sold in the month", ylabel="Units refunded (by order month)")
# Show the legend
ax.legend(title="Product", frameon=False)
# Save and show the figure
save_fig("fig13_sales_vs_refunds")

# ==============================================================================
# Interpretation
# - Refunds scale with sales (units sold vs units refunded: r = {...}), which is expected: a bigger product sells more and so is refunded more often in absolute numbers.
# - The refund rate, however, is not the same for every product: a chi-square test of independence is highly significant (chi-square = {...}, p < 0.001, Cramer's V = {...}, a small effect in statistical terms but commercially meaningful because refund rates range from {...}% to {...}%). Sugar Panda and Mr. Fuzzy have more refunds than expected, Love Bear and Mini Bear far fewer.
# - Selling more does not significantly worsen the refund rate (Spearman rho between monthly volume and refund rate = {...}, p = {...}), so the refund rate is driven by the *product* and by specific episodes (the Mr. Fuzzy spike circled in the chart), not by sales volume.
# - Cross-sell items are refunded at {...}% versus {...}% for primary items, so add-on sales are not causing extra returns. (This is partly because the most commonly added item, Mini Bear, has the lowest refund rate of all products.)
# ==============================================================================

# ==============================================================================
# ## 3.5 Hypothesis tests: are the differences real?
# We test the main differences found in Task 2 at a 5% significance level (alpha = 0.05). For comparing conversion rates we use a chi-square test on counts; for daily revenue, which is not normally distributed, we use the Mann-Whitney U test. Observational data cannot prove causation, but A/B tests (billing and landing pages) were run concurrently and carry stronger evidence.
# ==============================================================================

# List that will collect one row per hypothesis test
TESTS = []

def p_text(p):
    """Format a p-value for a table."""
    # Very small p-values are shown as "<0.001"
    return "<0.001" if p < 0.001 else f"{p:.3f}"

def two_prop_test(name, a_label, a_success, a_n, b_label, b_success, b_n):
    """Chi-square test for the difference between two proportions; adds a row to TESTS."""
    # Build the 2x2 table of successes and failures
    table = np.array([[a_success, a_n - a_success], [b_success, b_n - b_success]])
    # Run the chi-square test of independence
    res = stats.chi2_contingency(table)
    # Observed proportions
    pa, pb = a_success / a_n, b_success / b_n
    # Standard error of the difference (normal approximation)
    se = np.sqrt(pa * (1 - pa) / a_n + pb * (1 - pb) / b_n)
    # 95% confidence interval of the difference in percentage points
    lo, hi = (pa - pb - 1.96 * se) * 100, (pa - pb + 1.96 * se) * 100
    # Store the result
    TESTS.append({"hypothesis": name, "test": "Chi-square (2 proportions)", "statistic": res.statistic, "p_value": p_text(res.pvalue),
                  "effect": f"{a_label} {pa*100:.1f}% vs {b_label} {pb*100:.1f}% (diff {100*(pa-pb):+.1f} pp, 95% CI {lo:+.1f} to {hi:+.1f})",
                  "significant": "Yes" if res.pvalue < 0.05 else "No"})
    # Return the raw p-value for later use
    return res.pvalue

# Desktop vs mobile conversion
dsk, mob = sessions[sessions["device_type"] == "Desktop"], sessions[sessions["device_type"] == "Mobile"]
# Test 1
p_dev = two_prop_test("Desktop and mobile convert equally", "Desktop", dsk["converted"].sum(), len(dsk), "Mobile", mob["converted"].sum(), len(mob))
# New vs returning conversion (all traffic)
new_s, ret_s = sessions[sessions["visitor_type"] == "New"], sessions[sessions["visitor_type"] == "Returning"]
# Test 2
p_vis = two_prop_test("New and returning visitors convert equally (all traffic)", "Returning", ret_s["converted"].sum(), len(ret_s), "New", new_s["converted"].sum(), len(new_s))
# New vs returning inside the same brand-aware channels
sh_new, sh_ret = shared[shared["visitor_type"] == "New"], shared[shared["visitor_type"] == "Returning"]
# Test 3
p_vis2 = two_prop_test("New and returning convert equally (same channels only)", "Returning", sh_ret["converted"].sum(), len(sh_ret), "New", sh_new["converted"].sum(), len(sh_new))
# Google vs Bing non-brand conversion
g_s, b_s = sessions[sessions["channel"] == "Google nonbrand"], sessions[sessions["channel"] == "Bing nonbrand"]
# Test 4
p_gb = two_prop_test("Google and Bing non-brand convert equally", "Bing", b_s["converted"].sum(), len(b_s), "Google", g_s["converted"].sum(), len(g_s))
# Socialbook vs Google non-brand conversion
so_s = sessions[sessions["channel"] == "Social (Socialbook)"]
# Test 5
p_soc = two_prop_test("Socialbook and Google non-brand convert equally", "Socialbook", so_s["converted"].sum(), len(so_s), "Google", g_s["converted"].sum(), len(g_s))
# Billing page A/B test (concurrent period)
b1, b2 = bill_tbl.set_index("pageview_url").loc["/billing"], bill_tbl.set_index("pageview_url").loc["/billing-2"]
# Test 6
p_bill = two_prop_test("Billing page versions complete orders equally (A/B)", "/billing-2", b2["orders"], b2["sessions"], "/billing", b1["orders"], b1["sessions"])
# Landing page A/B test (concurrent period)
l2, l5 = lander_ab.set_index("landing_page").loc["/lander-2"], lander_ab.set_index("landing_page").loc["/lander-5"]
# Test 7
p_land = two_prop_test("/lander-2 and /lander-5 convert equally (desktop, concurrent)", "/lander-5", l5["orders"], l5["sessions"], "/lander-2", l2["orders"], l2["sessions"])
# Mr. Fuzzy refund spike versus the surrounding baseline
fz = items[(items["short_name"] == "Mr. Fuzzy") & (items["month"] >= "2014-01-01") & (items["month"] <= "2015-02-01")]
# Items inside the spike months
fz_spike, fz_base = fz[fz["month"].isin(spike.index)], fz[~fz["month"].isin(spike.index)]
# Test 8
p_spike = two_prop_test("Mr. Fuzzy refund rate in Aug-Sep 2014 equals other months", "Aug-Sep 2014", fz_spike["is_refunded"].sum(), len(fz_spike), "other months", fz_base["is_refunded"].sum(), len(fz_base))
# Cross-sell vs primary item refund rate
pr, cs = post_items[post_items["item_role"] == "Primary item"], post_items[post_items["item_role"] == "Cross-sell item"]
# Test 9
p_role = two_prop_test("Cross-sell and primary items are refunded equally", "Cross-sell", cs["is_refunded"].sum(), len(cs), "Primary", pr["is_refunded"].sum(), len(pr))

# Test 10: conversion rate differs across the six channels (chi-square on a 6 x 2 table)
ch_table = sessions.groupby("channel")["converted"].agg(["sum", "count"])
# Build the observed table: orders and non-orders per channel
obs = np.column_stack([ch_table["sum"], ch_table["count"] - ch_table["sum"]])
# Run the test
res_ch = stats.chi2_contingency(obs)
# Store the result
TESTS.append({"hypothesis": "All six channels convert equally", "test": "Chi-square (6 x 2)", "statistic": res_ch.statistic, "p_value": p_text(res_ch.pvalue),
              "effect": f"CVR from {ch_table['sum'].div(ch_table['count']).min()*100:.1f}% to {ch_table['sum'].div(ch_table['count']).max()*100:.1f}%", "significant": "Yes" if res_ch.pvalue < 0.05 else "No"})
# Test 11: product refund rates are equal (from Section 3.4)
TESTS.append({"hypothesis": "All four products have the same refund rate", "test": "Chi-square (4 x 2)", "statistic": chi.statistic, "p_value": p_text(chi.pvalue),
              "effect": f"Refund rate from {prod['refund_rate_pct'].min():.1f}% to {prod['refund_rate_pct'].max():.1f}% (Cramer's V = {cramers_v:.2f})", "significant": "Yes" if chi.pvalue < 0.05 else "No"})
# Test 12: weekday vs weekend daily revenue in 2014 (Mann-Whitney)
TESTS.append({"hypothesis": "Weekday and weekend daily revenue are equal (2014)", "test": "Mann-Whitney U", "statistic": mw_week.statistic, "p_value": p_text(mw_week.pvalue),
              "effect": f"Weekday {money(wd.mean())}/day vs weekend {money(we.mean())}/day", "significant": "Yes" if mw_week.pvalue < 0.05 else "No"})
# Test 13: Nov-Dec 2014 daily revenue higher than Sep-Oct 2014 (Mann-Whitney)
mw_hol = stats.mannwhitneyu(period(2014, [11, 12]), period(2014, [9, 10]), alternative="two-sided")
# Store the result
TESTS.append({"hypothesis": "Nov-Dec and Sep-Oct daily revenue are equal (2014)", "test": "Mann-Whitney U", "statistic": mw_hol.statistic, "p_value": p_text(mw_hol.pvalue),
              "effect": f"Nov-Dec {money(season.loc[2014, 'nov_dec_daily_rev'])}/day vs Sep-Oct {money(season.loc[2014, 'sep_oct_daily_rev'])}/day", "significant": "Yes" if mw_hol.pvalue < 0.05 else "No"})
# Test 14: no linear relationship between daily sessions and orders (Pearson)
TESTS.append({"hypothesis": "Daily sessions and orders are uncorrelated", "test": "Pearson correlation", "statistic": pear.statistic, "p_value": p_text(pear.pvalue),
              "effect": f"r = {pear.statistic:.2f}; {reg.slope*100:.1f} extra orders per 100 sessions", "significant": "Yes" if pear.pvalue < 0.05 else "No"})
# Convert the list of tests into a DataFrame
tests = pd.DataFrame(TESTS)
# Show every column completely (the effect text is long)
with pd.option_context("display.max_colwidth", 120):
    # Display the hypothesis-test summary
    display(tests)
# Store the key results for the narrative
M.update(p_gb=p_gb, p_vis2=p_vis2, p_land=p_land, p_bill=p_bill, p_soc=p_soc, p_role=p_role, p_spike=p_spike,
         n_tests=len(tests), n_sig=int((tests["significant"] == "Yes").sum()))

# ==============================================================================
# Reading the test table
# - All {...} of {...} tests are statistically significant (p < 0.05): the differences found in Task 2 are real patterns and not random noise.
# - Size matters more than significance. With samples this large, even tiny differences become "significant", so the effect column is what counts. Large, commercially important effects: desktop vs mobile (+5.4 pp), the new billing page (+17 pp), Socialbook vs Google (-3.4 pp), the Mr. Fuzzy refund spike (+9.8 pp). A small effect: Bing vs Google non-brand is detectable (p = {...}) but only a fraction of a percentage point, so the two search engines are close in quality.
# - Returning visitors keep a real, modest edge even inside the same channels (p = {...}), so it is not only a channel-mix artefact.
# - The billing page and landing page results come from tests where both versions ran at the same time with an almost even traffic split, so they are the most trustworthy evidence of cause and effect in this analysis.
# ==============================================================================

# ==============================================================================
# ## 3.6 Visualisations created
# The deliverable requires at least six meaningful visualisations. The notebook produced the following charts, all saved as PNG files in the figures folder:
# ==============================================================================

# List every chart file created in the figures folder
figure_files = sorted(f for f in os.listdir(FIG_DIR) if f.endswith(".png"))
# Print the number of charts
print(f"{len(figure_files)} visualisations saved:")
# Print each file name
for f in figure_files:
    # One chart per line
    print("  -", f)
# Verify that the minimum of six visualisations is met
assert len(figure_files) >= 6, "Fewer than six visualisations were created"
# Store the chart count
M["n_figures"] = len(figure_files)

# ==============================================================================
# # Task 4: Interpretation and Recommendations
# Here we combine the evidence into seven business insights. To judge how much each opportunity might be worth, we first compute simple *what-if sensitivities* from the data. They are illustrations based on 2014 volumes, not forecasts.
# ==============================================================================

# Sessions and orders of the last complete year (2014) used as the volume base for the what-if calculations
s14 = sessions[sessions["year"] == 2014]
# Number of orders in 2014
o14 = (orders["year"] == 2014).sum()
# 1) Each +1 percentage point of MOBILE conversion, on 2014 mobile sessions, at the mobile average order value
mobile_1pp = (s14["device_type"] == "Mobile").sum() * 0.01 * M["mobile_aov"]
# 2) Each +1 percentage point of ALL-traffic conversion on 2014 sessions at the 2014 average order value
overall_1pp = len(s14) * 0.01 * M["aov_2014"]
# 3) Each +1 percentage point of orders that contain a second item (extra value of a 2-item order over a 1-item order)
xsell_1pp = o14 * 0.01 * (M["aov_double"] - M["aov_single"])
# 4) Each +1 percentage point of customers who buy a second time, at the average order value
retention_1pp = M["customers"] * 0.01 * M["aov_total"]
# 5) Revenue-per-session gap between Socialbook and Google non-brand, applied to 10,000 sessions
social_gap_10k = (M["g_nb_rps"] - M["social_rps"]) * 10000
# 6) Sugar Panda refunds if its refund rate matched Love Bear's (the next-lowest-volume premium product)
panda_saving = prod.set_index("short_name").loc["Sugar Panda", "refunds"] * (1 - M["love_refund"] / M["panda_refund"])
# Collect the sensitivities in one table
sens = pd.DataFrame([
    ["Mobile conversion +1 pp", "2014 mobile sessions x 1% x mobile AOV", mobile_1pp, "per year"],
    ["Overall conversion +1 pp", "2014 sessions x 1% x average order value", overall_1pp, "per year"],
    ["Cross-sell rate +1 pp", "2014 orders x 1% x (2-item AOV - 1-item AOV)", xsell_1pp, "per year"],
    ["Repeat buyers +1 pp", "all customers x 1% x average order value", retention_1pp, "over 3 years"],
    ["Socialbook quality = Google non-brand", "revenue-per-session gap x 10,000 sessions", social_gap_10k, "per 10,000 sessions"],
    ["Sugar Panda refund rate = Love Bear rate", "Sugar Panda refund dollars x share avoided", panda_saving, "over 3 years"],
], columns=["lever", "assumption", "revenue_or_saving_usd", "period"])
# Display the what-if table
display(sens)
# Store the values for the narrative
M.update(mobile_1pp=mobile_1pp, overall_1pp=overall_1pp, xsell_1pp=xsell_1pp, retention_1pp=retention_1pp, social_gap_10k=social_gap_10k, panda_saving=panda_saving,
         mini_attach=attach.loc["Mini Bear", "cross_sell_rate_pct"] / 100, love_attach=attach.loc["Love Bear", "cross_sell_rate_pct"] / 100)

# ==============================================================================
# ## 4.1 Seven evidence-based business insights
# ### Insight 1 - Growth is strong, traffic-driven and seasonal
# - Evidence: gross revenue grew {...} in 2013 and {...} in 2014 (to {...}), and early-2015 revenue is still {...} versus the same period of 2014. Growth came from both more traffic (sessions: {...} to {...}) and better conversion ({...}% to {...}%). Daily sessions explain {...} of the variation in daily orders. November-December revenue per day runs {...} above September-October, and a weekend day earns only {...} of a weekday.
# - Recommendation: plan stock, customer support and ad budgets for the Q4 peak from September; test weekend-specific offers or emails to lift the weaker days.
# ### Insight 2 - Cross-selling is the cheapest way to raise order value
# - Evidence: since the second-item option launched ({...}), {...} of 2014 orders contained two items, worth {...} versus {...} for one item, which lifted average order value from {...} to {...}. Attach rates differ widely by the primary product: {...} for {...}, but only {...} for Love Bear and {...} for Mini Bear. Cross-sell items are not returned more often ({...}% refund rate vs {...}%).
# ==============================================================================

# ==============================================================================
# ## 4.4 Save the key results
# The final cell saves the key numbers and summary tables so that the report and presentation use exactly the same figures as this notebook.
# ==============================================================================

# Folder where the summary tables will be saved
TABLE_DIR = os.environ.get("TABLE_DIR", "tables")
# Create the folder if needed
os.makedirs(TABLE_DIR, exist_ok=True)
# Dictionary of the summary tables to export as CSV files
export = {"monthly_revenue": monthly, "quarterly_revenue": qtr, "yearly_summary": yearly.reset_index(), "product_performance": prod,
          "channel_performance": channel, "device_performance": device, "visitor_type_performance": visitor,
          "conversion_funnel": funnel.reset_index(), "refunds_by_product": refund_tbl, "hypothesis_tests": tests, "what_if_sensitivities": sens}
# Write each table to its own CSV file
for name, table in export.items():
    # Save without the row index
    table.to_csv(os.path.join(TABLE_DIR, f"{name}.csv"), index=False)
# Convert the key-number dictionary into plain JSON-friendly values
def to_plain(v):
    """Convert numpy/pandas values to plain Python types for JSON."""
    # Numpy numbers become Python floats or ints
    if isinstance(v, (np.integer,)):
        # Integers
        return int(v)
    # Numpy floats become Python floats
    if isinstance(v, (np.floating,)):
        # Floats
        return float(v)
    # Everything else (strings, timestamps) is stored as text
    return v if isinstance(v, (int, float, str, bool)) else str(v)
# Save the dictionary of key numbers
with open(os.path.join(TABLE_DIR, "key_metrics.json"), "w") as fh:
    # Write the dictionary as formatted JSON
    json.dump({k: to_plain(v) for k, v in M.items()}, fh, indent=2)
# Confirm what was saved
print(f"Saved {len(export)} tables and {len(M)} key metrics to '{TABLE_DIR}/'. Analysis complete.")
