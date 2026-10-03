import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import pickle
import re
import os
import plotly.express as px
import plotly.graph_objects as go

# ─── PAGE CONFIGURATION ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Customer Intelligence | E-Commerce",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── PATHS SETUP ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "ecommerce_hackathon.db")
CHURN_MODEL_PATH = os.path.join(BASE_DIR, "churn_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
MODEL_COLS_PATH = os.path.join(BASE_DIR, "model_columns.pkl")
SENTIMENT_MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        color: #f8fafc;
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 4px;
    }
    .status-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-positive { background-color: #065f46; color: #34d399; border: 1px solid #059669; }
    .badge-neutral  { background-color: #78350f; color: #fbbf24; border: 1px solid #d97706; }
    .badge-negative { background-color: #881337; color: #f43f5e; border: 1px solid #e11d48; }
    .badge-low      { background-color: #065f46; color: #34d399; }
    .badge-medium   { background-color: #78350f; color: #fbbf24; }
    .badge-high     { background-color: #881337; color: #f43f5e; }
</style>
""", unsafe_allow_html=True)


# ─── DATABASE HELPER ─────────────────────────────────────────────────────────
@st.cache_resource
def get_connection():
    if not os.path.exists(DB_PATH):
        st.error(f"Database not found at `{DB_PATH}`. Please ensure `ecommerce_hackathon.db` is present.")
        st.stop()
    return sqlite3.connect(DB_PATH, check_same_thread=False)

@st.cache_data(ttl=600)
def run_query(query: str) -> pd.DataFrame:
    conn = get_connection()
    return pd.read_sql_query(query, conn)


# ─── MODEL LOADERS ───────────────────────────────────────────────────────────
@st.cache_resource
def load_churn_artifacts():
    churn_model = pickle.load(open(CHURN_MODEL_PATH, "rb")) if os.path.exists(CHURN_MODEL_PATH) else None
    scaler = pickle.load(open(SCALER_PATH, "rb")) if os.path.exists(SCALER_PATH) else None
    col_info = pickle.load(open(MODEL_COLS_PATH, "rb")) if os.path.exists(MODEL_COLS_PATH) else None
    return churn_model, scaler, col_info

@st.cache_resource
def load_sentiment_artifacts():
    sentiment_model = pickle.load(open(SENTIMENT_MODEL_PATH, "rb")) if os.path.exists(SENTIMENT_MODEL_PATH) else None
    vectorizer = pickle.load(open(TFIDF_PATH, "rb")) if os.path.exists(TFIDF_PATH) else None
    return sentiment_model, vectorizer


# ─── SIDEBAR NAVIGATION ──────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", width=64)
    st.title("E-Commerce Intelligence")
    st.caption("AI-Powered Customer Analytics System")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["📊 Executive Dashboard", "🔮 Customer Churn Prediction", "💬 Review Sentiment Analysis"],
        index=0
    )
    st.markdown("---")
    st.info(
        "💡 **Tech Stack**\n\n"
        "- **Database:** SQLite (`ecommerce_hackathon.db`)\n"
        "- **Churn ML:** Random Forest Classifier\n"
        "- **Sentiment NLP:** TF-IDF + Logistic Regression\n"
        "- **Framework:** Streamlit + Plotly"
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1: EXECUTIVE DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
if page == "📊 Executive Dashboard":
    st.title("📊 E-Commerce Executive Dashboard")
    st.markdown("Real-time business performance analytics directly queried from the live database.")

    # 1. SQL KPIs
    kpi_query = """
    SELECT 
        ROUND(SUM(quantity * unit_price * (1.0 - discount)), 2) AS total_revenue,
        COUNT(order_id) AS total_orders,
        COUNT(DISTINCT customer_id) AS active_customers,
        ROUND(AVG(quantity * unit_price * (1.0 - discount)), 2) AS aov,
        ROUND(100.0 * SUM(CASE WHEN returned = 1 THEN 1 ELSE 0 END) / COUNT(order_id), 2) AS return_rate
    FROM orders;
    """
    kpis = run_query(kpi_query).iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Net Revenue</div>
            <div class="metric-value">${kpis['total_revenue']:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Orders</div>
            <div class="metric-value">{int(kpis['total_orders']):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Active Customers</div>
            <div class="metric-value">{int(kpis['active_customers']):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Order Value</div>
            <div class="metric-value">${kpis['aov']:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Overall Return Rate</div>
            <div class="metric-value">{kpis['return_rate']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 2. Charts Row 1: Monthly Trend & Category Revenue
    row1_col1, row1_col2 = st.columns([6, 5])

    with row1_col1:
        st.subheader("📈 Monthly Net Revenue & Order Volume")
        monthly_query = """
        SELECT 
            STRFTIME('%Y-%m', order_date) AS order_month,
            COUNT(order_id) AS order_count,
            ROUND(SUM(quantity * unit_price * (1.0 - discount)), 2) AS monthly_revenue
        FROM orders
        WHERE order_date IS NOT NULL
        GROUP BY order_month
        ORDER BY order_month ASC;
        """
        df_monthly = run_query(monthly_query)

        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Bar(
            x=df_monthly['order_month'],
            y=df_monthly['order_count'],
            name="Order Count",
            marker_color="#334155",
            opacity=0.6,
            yaxis="y2"
        ))
        fig_monthly.add_trace(go.Scatter(
            x=df_monthly['order_month'],
            y=df_monthly['monthly_revenue'],
            name="Net Revenue ($)",
            mode="lines+markers",
            line=dict(color="#38bdf8", width=3),
            marker=dict(size=6, color="#0284c7")
        ))
        fig_monthly.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(title="Net Revenue ($)", side="left"),
            yaxis2=dict(title="Orders Count", overlaying="y", side="right", showgrid=False)
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

    with row1_col2:
        st.subheader("🏷️ Category Revenue Performance")
        category_query = """
        SELECT 
            p.category,
            ROUND(SUM(o.quantity * o.unit_price * (1.0 - o.discount)), 2) AS net_revenue,
            COUNT(o.order_id) AS orders_count,
            ROUND(100.0 * SUM(CASE WHEN o.returned = 1 THEN 1 ELSE 0 END) / COUNT(o.order_id), 2) AS return_rate
        FROM products p
        JOIN orders o ON p.product_id = o.product_id
        GROUP BY p.category
        ORDER BY net_revenue ASC;
        """
        df_cat = run_query(category_query)

        fig_cat = px.bar(
            df_cat,
            x="net_revenue",
            y="category",
            orientation="h",
            color="net_revenue",
            color_continuous_scale="Tealgrn",
            labels={"net_revenue": "Net Revenue ($)", "category": "Product Category"}
        )
        fig_cat.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    # 3. Charts Row 2: City Revenue & Return Rate by Category
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.subheader("🏙️ Top Cities by Revenue")
        city_query = """
        SELECT 
            c.city,
            ROUND(SUM(o.quantity * o.unit_price * (1.0 - o.discount)), 2) AS city_revenue,
            COUNT(DISTINCT c.customer_id) AS customer_count
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        GROUP BY c.city
        ORDER BY city_revenue DESC
        LIMIT 8;
        """
        df_city = run_query(city_query)
        fig_city = px.bar(
            df_city,
            x="city",
            y="city_revenue",
            color="city_revenue",
            color_continuous_scale="Purples",
            labels={"city_revenue": "Revenue ($)", "city": "City"}
        )
        fig_city.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=30, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig_city, use_container_width=True)

    with row2_col2:
        st.subheader("↩️ Return Rate by Product Category")
        fig_ret = px.bar(
            df_cat.sort_values("return_rate", ascending=False),
            x="category",
            y="return_rate",
            color="return_rate",
            color_continuous_scale="Reds",
            labels={"return_rate": "Return Rate (%)", "category": "Category"}
        )
        fig_ret.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=30, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig_ret, use_container_width=True)

    # 4. Tables Row: Top 5 Products & Top 10 Spending Customers
    st.markdown("---")
    t1, t2 = st.columns(2)

    with t1:
        st.subheader("🏆 Top 5 Products by Revenue")
        top_prod_query = """
        SELECT 
            p.product_id AS ID,
            p.name AS Product,
            p.category AS Category,
            p.brand AS Brand,
            SUM(o.quantity) AS Units,
            ROUND(SUM(o.quantity * o.unit_price * (1.0 - o.discount)), 2) AS Revenue
        FROM products p
        JOIN orders o ON p.product_id = o.product_id
        GROUP BY p.product_id, p.name, p.category, p.brand
        ORDER BY Revenue DESC
        LIMIT 5;
        """
        df_top_prod = run_query(top_prod_query)
        df_top_prod['Revenue'] = df_top_prod['Revenue'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(df_top_prod, use_container_width=True, hide_index=True)

    with t2:
        st.subheader("💎 Top 10 VIP Customers by Spending")
        top_cust_query = """
        SELECT 
            c.name AS Customer,
            c.city AS City,
            c.membership_type AS Membership,
            COUNT(o.order_id) AS Orders,
            ROUND(SUM(o.quantity * o.unit_price * (1.0 - o.discount)), 2) AS Total_Spent
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        GROUP BY c.customer_id, c.name, c.city, c.membership_type
        ORDER BY Total_Spent DESC
        LIMIT 10;
        """
        df_top_cust = run_query(top_cust_query)
        df_top_cust['Total_Spent'] = df_top_cust['Total_Spent'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(df_top_cust, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2: CUSTOMER CHURN PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Customer Churn Prediction":
    st.title("🔮 Customer Churn Prediction")
    st.markdown("Predict whether a customer is at risk of churning using trained Machine Learning models.")

    churn_model, scaler, col_info = load_churn_artifacts()

    if churn_model is None or col_info is None:
        st.warning("⚠️ Churn model artifacts (`churn_model.pkl`, `model_columns.pkl`) were not found. Please run the notebook first.")
    else:
        st.success(f"✅ Loaded ML Model: **{type(churn_model).__name__}**")

        st.subheader("📋 Customer Profile & Behavioral Attributes")
        col_a, col_b = st.columns(2)

        with col_a:
            age = st.slider("Customer Age", min_value=18, max_value=85, value=35, help="Age of the customer")
            membership = st.selectbox("Membership Tier", ["Bronze", "Silver", "Gold"], index=1)
            total_orders = st.number_input("Total Historical Orders", min_value=1, max_value=200, value=8)
            total_spending = st.number_input("Total Net Spending ($)", min_value=10.0, max_value=100000.0, value=2450.0, step=50.0)

        with col_b:
            days_since_last_order = st.slider("Days Since Last Order", min_value=1, max_value=365, value=45, help="Recency of last purchase")
            return_rate = st.slider("Return Rate (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.5) / 100.0
            avg_delivery_days = st.slider("Average Delivery Time (Days)", min_value=1.0, max_value=30.0, value=4.5, step=0.5)

        avg_order_value = total_spending / total_orders
        st.info(f"💡 Calculated **Average Order Value (AOV):** `${avg_order_value:.2f}`")

        # Encode membership: Bronze=1, Silver=2, Gold=3
        membership_map = {"Bronze": 1, "Silver": 2, "Gold": 3}
        membership_encoded = membership_map.get(membership, 1)

        if st.button("🚀 Analyze Churn Risk", type="primary", use_container_width=True):
            # Prepare feature vector matching training order
            input_dict = {
                'total_orders': total_orders,
                'total_spending': total_spending,
                'avg_order_value': avg_order_value,
                'days_since_last_order': days_since_last_order,
                'return_rate': return_rate,
                'avg_delivery_days': avg_delivery_days,
                'age': age,
                'membership_encoded': membership_encoded
            }

            feature_cols = col_info.get('columns', list(input_dict.keys()))
            input_df = pd.DataFrame([input_dict])[feature_cols]

            if col_info.get('uses_scaler', False) and scaler is not None:
                features_ready = scaler.transform(input_df)
            else:
                features_ready = input_df

            # Predict probability
            churn_proba = churn_model.predict_proba(features_ready)[0, 1]
            churn_pred = int(churn_proba >= 0.5)

            st.markdown("---")
            st.subheader("🎯 Prediction Results")

            res_col1, res_col2 = st.columns([5, 5])

            with res_col1:
                st.metric("Churn Probability", f"{churn_proba * 100:.1f}%")
                st.progress(float(churn_proba))

                if churn_proba >= 0.65:
                    st.markdown("""<div class="status-badge badge-high">⚠️ HIGH CHURN RISK</div>""", unsafe_allow_html=True)
                    st.error("This customer exhibits critical signals of attrition. Immediate proactive retention is required.")
                elif churn_proba >= 0.35:
                    st.markdown("""<div class="status-badge badge-medium">⚠️ MEDIUM CHURN RISK</div>""", unsafe_allow_html=True)
                    st.warning("Customer behavior indicates cooling engagement. Recommend nurturing campaigns.")
                else:
                    st.markdown("""<div class="status-badge badge-low">✅ LOW CHURN RISK (RETAINED)</div>""", unsafe_allow_html=True)
                    st.success("Customer is actively engaged with healthy recency and spend patterns.")

            with res_col2:
                st.markdown("#### 🛡️ Recommended Retention Strategy")
                if churn_proba >= 0.65:
                    st.markdown("""
                    - **Personalized Win-Back Offer:** Send an automated 15-20% limited-time incentive.
                    - **Customer Success Outreach:** Investigate potential friction (e.g. delivery delays or return issues).
                    - **Exclusive VIP Perk:** Grant temporary Gold membership tier access to re-anchor brand value.
                    """)
                elif churn_proba >= 0.35:
                    st.markdown("""
                    - **Category Re-Engagement:** Highlight top-rated arrivals in their preferred category.
                    - **Loyalty Points Reminder:** Nudge points balance before expiration.
                    - **Survey Trigger:** Collect feedback regarding their recent ordering experience.
                    """)
                else:
                    st.markdown("""
                    - **Upsell / Cross-Sell:** Feature complementary accessory items.
                    - **Referral Invitation:** Prompt advocacy with a 'Refer-a-Friend' program bonus.
                    - **Early Access Privilege:** Invite to upcoming flash sales and exclusive product drops.
                    """)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3: REVIEW SENTIMENT ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "💬 Review Sentiment Analysis":
    st.title("💬 Review Sentiment Analysis")
    st.markdown("Automated natural language processing to extract sentiment and customer feedback sentiment.")

    sent_model, tfidf_vec = load_sentiment_artifacts()

    if sent_model is None or tfidf_vec is None:
        st.warning("⚠️ Sentiment artifacts (`sentiment_model.pkl`, `tfidf_vectorizer.pkl`) not found. Please run the notebook first.")
    else:
        st.success("✅ NLP Model Ready: **TF-IDF + Logistic Regression Classifier**")

        st.subheader("📝 Input Customer Review")

        # Preset examples for convenience
        preset = st.selectbox(
            "Quick Demo Examples (Select to test):",
            [
                "Custom Text...",
                "The product exceeded all my expectations! Ultra fast shipping and top quality build.",
                "Disappointed. The item stopped working after two days and customer support never replied.",
                "It is okay for the price. Delivery was acceptable but packaging was slightly crumpled.",
                "Absolutely brilliant customer service, solved my issue in minutes and great warranty!",
                "Terrible experience, arrived damaged and returning it was a complete nightmare."
            ]
        )

        default_text = "" if preset == "Custom Text..." else preset
        review_input = st.text_area(
            "Enter customer review text below:",
            value=default_text,
            height=120,
            placeholder="Type or paste any product review here..."
        )

        if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
            if not review_input.strip():
                st.warning("Please enter review text to analyze.")
            else:
                # Text cleaning
                cleaned = str(review_input).lower()
                cleaned = re.sub(r'[^a-z\s]', ' ', cleaned)
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()

                # Vectorize & Predict
                X_vec = tfidf_vec.transform([cleaned])
                pred_label = sent_model.predict(X_vec)[0]
                proba = sent_model.predict_proba(X_vec)[0]
                classes = list(sent_model.classes_)

                st.markdown("---")
                st.subheader("🎯 Sentiment Classification")

                s1, s2 = st.columns([4, 6])

                with s1:
                    if pred_label == "Positive":
                        st.markdown("""<div class="status-badge badge-positive" style="font-size:1.2rem;">🌟 POSITIVE SENTIMENT</div>""", unsafe_allow_html=True)
                        st.balloons()
                    elif pred_label == "Neutral":
                        st.markdown("""<div class="status-badge badge-neutral" style="font-size:1.2rem;">⚖️ NEUTRAL SENTIMENT</div>""", unsafe_allow_html=True)
                    else:
                        st.markdown("""<div class="status-badge badge-negative" style="font-size:1.2rem;">⚠️ NEGATIVE SENTIMENT</div>""", unsafe_allow_html=True)

                    confidence = max(proba) * 100
                    st.metric("Model Confidence", f"{confidence:.1f}%")
                    st.caption(f"Cleaned tokens: `{cleaned}`")

                with s2:
                    df_probs = pd.DataFrame({
                        "Sentiment": classes,
                        "Probability": proba * 100
                    })
                    color_map = {"Positive": "#34d399", "Neutral": "#fbbf24", "Negative": "#f43f5e"}
                    fig_sent = px.bar(
                        df_probs,
                        x="Probability",
                        y="Sentiment",
                        orientation="h",
                        color="Sentiment",
                        color_discrete_map=color_map,
                        text=df_probs["Probability"].apply(lambda p: f"{p:.1f}%")
                    )
                    fig_sent.update_layout(
                        template="plotly_dark",
                        margin=dict(l=20, r=20, t=20, b=20),
                        showlegend=False,
                        xaxis=dict(range=[0, 100], title="Probability (%)")
                    )
                    st.plotly_chart(fig_sent, use_container_width=True)
