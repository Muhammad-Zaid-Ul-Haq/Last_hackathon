import re, sqlite3
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent                      # relative paths so it works on Streamlit Cloud
DB_PATH = next((p for p in (BASE / "ecommerce_clean.db", BASE / "ecommerce_hackathon.db") if p.is_file()), BASE / "ecommerce_hackathon.db")

# column names (must match the notebook's config)
QTY, PRICE, DISC, DATE = "quantity", "unit_price", "discount", "order_date"
NET = f"{QTY} * {PRICE} * (1 - {DISC})"

st.set_page_config(page_title="E-Commerce Customer Intelligence", layout="wide")

# ---------------------------------------------------------------- startup checks
REQUIRED_FILES = ["churn_model.pkl", "scaler.pkl", "model_columns.pkl",
                  "sentiment_model.pkl", "tfidf_vectorizer.pkl"]
REQUIRED_COLS = {
    "orders": ["order_id", "customer_id", "product_id", DATE, QTY, PRICE, DISC],
    "customers": ["customer_id"],
    "products": ["product_id", "category"],
}


def startup_problems():
    problems = []
    if not DB_PATH.exists():
        problems.append("Database not found: put `ecommerce_clean.db` (or `ecommerce_hackathon.db`) "
                        f"in `{BASE}`.")
    else:
        with sqlite3.connect(DB_PATH) as conn:
            for table, cols in REQUIRED_COLS.items():
                have = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
                if not have:
                    problems.append(f"Table `{table}` not found in {DB_PATH.name}.")
                    continue
                miss = [c for c in cols if c not in have]
                if miss:
                    problems.append(f"Table `{table}` has columns {have}, but app.py expects {miss}. "
                                    "Edit the column constants at the top of app.py.")
    for f in REQUIRED_FILES:
        if not (BASE / f).exists():
            problems.append(f"Missing model file `{f}` in `{BASE}` - run the notebook (Tasks D and F) first.")
    return problems


_problems = startup_problems()
if _problems:
    st.title("Setup problem")
    for p in _problems:
        st.error(p)
    st.info(f"app.py is running from: {BASE}")
    st.stop()


@st.cache_data
def run_query(sql: str) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(sql, conn)


@st.cache_resource
def load_churn():  # a pickle made with another scikit-learn version fails here
    model = joblib.load(BASE / "churn_model.pkl")
    scaler = joblib.load(BASE / "scaler.pkl")
    metadata = joblib.load(BASE / "model_columns.pkl")
    # Newer notebook output stores feature order and whether scaling was used.
    # Accept a plain list too, for older artifact versions.
    columns = metadata.get("columns", metadata) if isinstance(metadata, dict) else metadata
    uses_scaler = metadata.get("uses_scaler", True) if isinstance(metadata, dict) else True
    return model, scaler, list(columns), bool(uses_scaler)


@st.cache_resource
def load_sentiment():
    return joblib.load(BASE / "sentiment_model.pkl"), joblib.load(BASE / "tfidf_vectorizer.pkl")


# Match preprocess_text() used to train the saved TF-IDF vectorizer.
def clean_text(t):
    t = re.sub(r"[^a-z\s]", " ", str(t).lower())
    return re.sub(r"\s+", " ", t).strip()


page = st.sidebar.radio("Navigation", ["Dashboard", "Churn Prediction", "Sentiment Analysis"])

# ---------------------------------------------------------------- Dashboard
if page == "Dashboard":
    st.title("📊 Business Dashboard")
    revenue = run_query(f"SELECT SUM({NET}) AS v FROM orders")["v"][0]
    n_orders = run_query("SELECT COUNT(*) AS v FROM orders")["v"][0]
    n_cust = run_query("SELECT COUNT(*) AS v FROM customers")["v"][0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Net Revenue", f"{revenue:,.0f}")
    c2.metric("Total Orders", f"{n_orders:,}")
    c3.metric("Total Customers", f"{n_cust:,}")

    monthly = run_query(f"SELECT STRFTIME('%Y-%m', {DATE}) AS month, SUM({NET}) AS net_revenue "
                        f"FROM orders GROUP BY month ORDER BY month")
    st.subheader("Monthly Net Revenue")
    st.line_chart(monthly.set_index("month"))

    cat = run_query(f"SELECT p.category, SUM(o.{QTY} * o.{PRICE} * (1 - o.{DISC})) AS net_revenue "
                    f"FROM orders o JOIN products p ON o.product_id = p.product_id "
                    f"GROUP BY p.category ORDER BY net_revenue DESC")
    st.subheader("Revenue by Category")
    st.bar_chart(cat.set_index("category"))

# ---------------------------------------------------------------- Churn
elif page == "Churn Prediction":
    st.title("🔮 Customer Churn Prediction")
    model, scaler, columns, uses_scaler = load_churn()
    c1, c2 = st.columns(2)
    age = c1.number_input("Age", 18, 100, 35)
    membership = c1.selectbox("Membership type", ["Bronze", "Silver", "Gold", "Standard"])
    total_orders = c1.number_input("Total orders", 1, 1000, 5)
    total_spending = c1.number_input("Total spending", 0.0, 1e7, 2000.0, step=100.0)
    days_since = c2.number_input("Days since last order", 0, 2000, 60)
    return_rate = c2.slider("Return rate", 0.0, 1.0, 0.1, 0.01)
    delivery = c2.number_input("Avg delivery days", 0.0, 60.0, 5.0, step=0.5)

    if st.button("Predict churn"):
        row = {"total_orders": total_orders, "total_spending": total_spending,
               "avg_order_value": total_spending / max(total_orders, 1),
               "days_since_last_order": days_since, "return_rate": return_rate,
               "avg_delivery_days": delivery, "age": age,
               "membership_encoded": {"Bronze": 1, "Silver": 2, "Gold": 3, "Standard": 1}[membership]}
        X = pd.DataFrame([row]).reindex(columns=columns, fill_value=0)  # exact training order
        model_input = scaler.transform(X) if uses_scaler else X
        prob = float(model.predict_proba(model_input)[0, list(model.classes_).index(1)])
        st.metric("Churn probability", f"{prob:.1%}")
        st.progress(prob)
        (st.error if prob >= 0.5 else st.success)(
            "⚠️ High churn risk" if prob >= 0.5 else "✅ Likely to stay")

# ---------------------------------------------------------------- Sentiment
else:
    st.title("💬 Review Sentiment Analysis")
    model, tfidf = load_sentiment()
    text = st.text_area("Enter a review", height=150)
    if st.button("Analyze") and text.strip():
        vec = tfidf.transform([clean_text(text)])
        probs = model.predict_proba(vec)[0]
        label = model.classes_[probs.argmax()]
        st.subheader(f"Predicted sentiment: {label}")
        st.metric("Confidence", f"{probs.max():.1%}")
        st.bar_chart(pd.Series(probs, index=model.classes_))
