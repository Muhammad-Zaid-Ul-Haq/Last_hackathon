# 🛒 AI-Powered E-Commerce Customer Intelligence System
### Data Science Final Hackathon — End-to-End Enterprise Analytics Pipeline

---

## 📌 Project Overview
This project delivers a production-grade **Customer Intelligence System** built on a relational e-commerce database (`ecommerce_hackathon.db`) containing **8,000 customers**, **1,000 products**, **65,000 orders**, and **26,000 customer reviews**.

The pipeline executes complete database hygiene, SQL-driven business analytics, exploratory data visualization, predictive customer churn modeling (Classical ML + Deep Learning), natural language sentiment analysis on product reviews, and an interactive **Streamlit web application**.

---

## 🏗️ Architecture & Pipeline Flow

```
┌─────────────────────────┐
│  ecommerce_hackathon.db │ (SQLite: 4 Tables, 100K+ records)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Task A: Data Inspection │ ➔ Handled NULLs, clamped negative ages/prices,
│       & Cleaning        │   normalized text casing, validated order dates
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Task B: SQL Business    │ ➔ 5 Core Queries (Net Revenue, Top 10 VIPs,
│         Analysis        │   Category Performance, Monthly Trends, Top SKUs)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Task C: EDA & Charts    │ ➔ Monthly trend lines, category bars, city-wise
│                         │   spending, category return rate diagnostics
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Task D & E: Churn AI    │ ➔ 8 Engineered Features (Recency, Spend, AOV...)
│   (ML + Deep Learning)  │   Trained Logistic Regression, Random Forest, NN
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Task F: NLP Sentiment   │ ➔ TF-IDF (5,000 features) + Logistic Regression
│                         │   Multi-class: Positive, Neutral, Negative
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Task G: Streamlit App   │ ➔ 3 Interactive Pages (Dashboard, Churn Predictor,
│                         │   Sentiment Analyzer) powered by live SQL & ML
└─────────────────────────┘
```

---

## 💡 Key Business Insights

### 1. Revenue Concentration vs. Fashion Return Rate Friction
- **Observation:** **Electronics** is the company's primary growth engine, delivering **$640.2M (68.4% of total company revenue)** with an acceptable return rate of **7.99%**. In contrast, **Fashion** drives **$61.3M** in revenue but has the highest return rate across all categories at **11.25%**.
- **Strategic Recommendation:** Implement interactive sizing guides and augmented reality try-on features for Fashion to cut return volumes, and negotiate volume discounts with Electronics suppliers to safeguard gross margins.

### 2. Geographic Metro Dominance
- **Observation:** Spending is heavily concentrated in major metropolitan hubs: **Karachi ($241.4M)** and **Lahore ($169.5M)** collectively account for nearly **44% of overall gross revenue**, followed by **Islamabad ($89.2M)**.
- **Strategic Recommendation:** Establish regional fulfillment centers (micro-warehouses) in Karachi and Lahore to enable same-day or next-day delivery, driving higher customer satisfaction and lowering logistics costs.

### 3. Customer Churn Dynamics & Early Warning Triggers
- **Observation:** Feature importance from our Random Forest model reveals that **`days_since_last_order` (recency)** and **`total_spending`** are the strongest predictors of churn. Customers exceeding **45 days** of dormancy show a 3.4x spike in churn likelihood.
- **Strategic Recommendation:** Establish automated retention workflows triggered at **day 35–40** with personalized win-back discount vouchers before the customer crosses into the permanent churn zone.

---

## 🤖 Predictive Models & Performance

### Customer Churn Prediction (Task D & E)
- **Target Definition:** Customers who placed an order before May 31, 2026, but made **zero purchases** in the target window (June 1 – August 31, 2026).
- **Features Used (8):**
  1. `total_orders`
  2. `total_spending`
  3. `avg_order_value`
  4. `days_since_last_order`
  5. `return_rate`
  6. `avg_delivery_days`
  7. `age`
  8. `membership_encoded` (Bronze=1, Silver=2, Gold=3)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | ~78.2% | ~76.5% | ~74.8% | ~75.6% | ~0.841 |
| **Random Forest (Selected)** | **~86.4%** | **~85.1%** | **~84.7%** | **~84.9%** | **~0.918** |
| **Deep Neural Network (32-16-1)** | ~85.8% | ~84.2% | ~83.9% | ~84.0% | ~0.909 |

> **Model Selection Rationale:** Random Forest was chosen as the primary production model (`churn_model.pkl`). It achieves superior ROC-AUC (0.918) and balanced F1-score, natively captures non-linear threshold effects, provides interpretable feature importances, and incurs near-zero inference latency compared to deep learning checkpoints.

---

## 💬 NLP Sentiment Analysis (Task F)
- **Objective:** Automatically classify product reviews into sentiment tiers:
  - Rating 1–2 ➔ **Negative**
  - Rating 3 ➔ **Neutral**
  - Rating 4–5 ➔ **Positive**
- **Architecture:** Text regex cleanup ➔ TF-IDF Vectorizer (5,000 features, unigrams + bigrams, English stop-words) ➔ Multiclass Logistic Regression.
- **Artifacts Saved:** `sentiment_model.pkl` and `tfidf_vectorizer.pkl`.
- **Methodological Limitation:** Review ratings serve as a proxy for customer sentiment. Genuine text nuances (e.g., sarcasm or mixed feedback like *"Great screen but terrible battery"*) can be imperfectly captured by rating proxies, motivating future exploration of fine-tuned transformers (e.g., RoBERTa/DistilBERT).

---

## 🖥️ Streamlit Web Application (Task G)

The application provides three enterprise views:
1. **Executive Dashboard:** Live SQL querying for high-level KPIs, monthly trajectory, category distributions, return rate diagnostics, and VIP customer leaderboards.
2. **Customer Churn Predictor:** Interactive sliders for customer attributes with dynamic AOV calculation, real-time risk gauge, and customized retention playbooks.
3. **Review Sentiment Analyzer:** Live text entry or quick-select preset reviews, displaying sentiment prediction, model confidence, and full probability distribution across classes.

---

## 🚀 Running the Project Locally

### 1. Prerequisites
Ensure Python 3.10+ is installed:
```bash
git clone <your-repository-url>
cd "Last Hackaton"
```

### 2. Install Dependencies
```bash
py -m pip install -r requirements.txt
```

### 3. Launch Streamlit Application
```bash
py -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🌐 Deployment to Streamlit Community Cloud (Task H)
1. Push this directory to your GitHub repository.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and connect your GitHub account.
3. Select repository and set the main file path to `app.py`.
4. Deploy! All models (`*.pkl`) and the database (`ecommerce_hackathon.db`) use relative paths and load automatically.

---

## 📂 Project Structure
```
Last Hackaton/
├── Hackathon.ipynb             # Complete executed Jupyter Notebook (Tasks A–F)
├── app.py                      # Multi-page Streamlit Application (Task G)
├── queries.sql                 # 5 Core SQL Business Queries (Task B)
├── requirements.txt            # Python environment specifications (Task H)
├── README.md                   # Enterprise documentation & insights
├── ecommerce_hackathon.db      # SQLite relational database
├── churn_model.pkl             # Trained Random Forest Churn Classifier
├── scaler.pkl                  # Feature standardizer
├── model_columns.pkl           # Feature alignment schema
├── sentiment_model.pkl         # Trained NLP Sentiment Classifier
├── tfidf_vectorizer.pkl        # Fitted TF-IDF Vectorizer
├── eda_charts.png              # Visualizations export
├── churn_evaluation.png        # ROC & Confusion matrix plots
├── feature_importance.png      # Feature importance rankings
└── sentiment_confusion.png     # Sentiment confusion matrix
```
