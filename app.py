import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Import exact functions and constants from your original script without altering them
from train_model import (
    MODEL_PATH,
    find_data_file,
    prepare_features,
    train_gradient_boosting_from_notebook,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Bank Fraud AI Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, professional UI
st.markdown(
    """
    <style>
    .main { background-color: #0F172A; }
    .stMetric {
        background-color: #1E293B;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #334155;
    }
    .stCard {
        background-color: #1E293B;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #334155;
        margin-bottom: 20px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Data & Model Loader Functions
# -----------------------------------------------------------------------------
@st.cache_data
def load_raw_data():
    """Load raw dataframe using original helper logic."""
    data_path = find_data_file()
    return pd.read_csv(data_path), data_path


@st.cache_resource
def load_or_train_model():
    """Load pre-trained pickle model or trigger exact training workflow."""
    if not MODEL_PATH.exists():
        train_gradient_boosting_from_notebook()

    with MODEL_PATH.open("rb") as fh:
        model = pickle.load(fh)
    return model


# Load Data and Model
try:
    df_raw, data_path = load_raw_data()
    X, y = prepare_features(df_raw)
    model = load_or_train_model()

    # Create standard train/test split matching your training routine
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Compute evaluation predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

except Exception as e:
    st.error(f"Error initializing dashboard components: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
st.sidebar.title("🛡️ Bank Fraud AI")
st.sidebar.caption("HistGradientBoosting Detection Engine")
st.sidebar.divider()

st.sidebar.subheader("Dataset Info")
st.sidebar.info(f"**Loaded File:**\n`{data_path.name}`")
st.sidebar.text(f"Total Rows: {len(df_raw):,}")
st.sidebar.text(f"Features Engineered: {X.shape[1]}")

st.sidebar.divider()
if st.sidebar.button("🔄 Retrain Model"):
    with st.spinner("Retraining HistGradientBoosting model..."):
        train_gradient_boosting_from_notebook()
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# -----------------------------------------------------------------------------
# Main Dashboard Layout
# -----------------------------------------------------------------------------
st.title("💳 Financial Fraud Analytics & Prediction Dashboard")
st.markdown(
    "Real-time monitoring, model metrics, and batch inference engine for financial transaction streams."
)

tabs = st.tabs(
    [
        "📊 Model Performance",
        "🔎 Transaction Inspector",
        "📈 Feature & Data Explorer",
    ]
)

# -----------------------------------------------------------------------------
# TAB 1: Model Performance
# -----------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Model Performance Summary")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    col1.metric("Precision", f"{prec:.4f}")
    col2.metric("Recall", f"{rec:.4f}")
    col3.metric("F1-Score", f"{f1:.4f}")
    col4.metric("ROC-AUC Score", f"{auc:.4f}")

    st.divider()

    m_col1, m_col2 = st.columns(2)

    with m_col1:
        st.markdown("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Legit", "Actual Fraud"],
            columns=["Pred Legit", "Pred Fraud"],
        )

        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            labels=dict(x="Predicted Label", y="True Label", color="Count"),
        )
        fig_cm.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_cm, use_container_width=True)

    with m_col2:
        st.markdown("### Classification Report")
        report = classification_report(
            y_test, y_pred, target_names=["Legitimate", "Fraudulent"], output_dict=True
        )
        report_df = pd.DataFrame(report).transpose().round(4)
        st.dataframe(report_df, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: Live Transaction Simulator / Predictor
# -----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Single Transaction Fraud Risk Assessment")
    st.markdown("Simulate input variables to compute live risk scores.")

    with st.form("prediction_form"):
        p_col1, p_col2, p_col3 = st.columns(3)

        with p_col1:
            amount = st.number_input("Transaction Amount ($)", value=10000.0, step=500.0)
            txn_type = st.selectbox(
                "Transaction Type",
                options=(
                    ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"]
                    if "type" in df_raw.columns
                    else ["TRANSFER"]
                ),
            )
            nameDest = st.text_input("Destination Name (e.g., M1234567)", value="C987654321")

        with p_col2:
            oldbalanceOrg = st.number_input("Origin Old Balance ($)", value=10000.0, step=500.0)
            newbalanceOrig = st.number_input("Origin New Balance ($)", value=0.0, step=500.0)

        with p_col3:
            oldbalanceDest = st.number_input("Destination Old Balance ($)", value=0.0, step=500.0)
            newbalanceDest = st.number_input("Destination New Balance ($)", value=10000.0, step=500.0)

        submit = st.form_submit_button("Assess Fraud Risk", use_container_width=True)

    if submit:
        # Build single observation dataframe matching raw format
        sample_dict = {
            "amount": [amount],
            "oldbalanceOrg": [oldbalanceOrg],
            "newbalanceOrig": [newbalanceOrig],
            "oldbalanceDest": [oldbalanceDest],
            "newbalanceDest": [newbalanceDest],
            "nameOrig": ["C123456789"],
            "nameDest": [nameDest],
            "type": [txn_type],
            "isFlaggedFraud": [0],
            "isFraud": [0],  # Dummy column needed for prepare_features pipeline
        }

        sample_df = pd.DataFrame(sample_dict)
        X_sample, _ = prepare_features(sample_df)

        # Align sample columns with training schema
        missing_cols = set(X.columns) - set(X_sample.columns)
        for col in missing_cols:
            X_sample[col] = 0
        X_sample = X_sample[X.columns]

        risk_score = model.predict_proba(X_sample)[0][1]
        is_fraud_pred = model.predict(X_sample)[0]

        st.divider()
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            if is_fraud_pred == 1:
                st.error("🚨 HIGH RISK: FRAUD DETECTED")
            else:
                st.success("✅ LOW RISK: TRANSACTION LEGITIMATE")

            st.metric("Fraud Probability Score", f"{risk_score * 100:.2f}%")

        with res_col2:
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=risk_score * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Fraud Risk Score Index"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#EF4444" if risk_score > 0.5 else "#10B981"},
                        "steps": [
                            {"range": [0, 30], "color": "#064E3B"},
                            {"range": [30, 70], "color": "#78350F"},
                            {"range": [70, 100], "color": "#7F1D1D"},
                        ],
                    },
                )
            )
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: Feature & Data Explorer
# -----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Dataset Overview & Engineered Feature Insights")

    st.markdown("### Raw Data Sample")
    st.dataframe(df_raw.head(10), use_container_width=True)

    e_col1, e_col2 = st.columns(2)

    with e_col1:
        st.markdown("### Class Balance (Raw)")
        if "isFraud" in df_raw.columns:
            class_counts = df_raw["isFraud"].value_counts().reset_index()
            class_counts.columns = ["Class", "Count"]
            class_counts["Class"] = class_counts["Class"].map({0: "Legit", 1: "Fraud"})

            fig_pie = px.pie(
                class_counts,
                names="Class",
                values="Count",
                color="Class",
                color_discrete_map={"Legit": "#3B82F6", "Fraud": "#EF4444"},
                hole=0.4,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with e_col2:
        st.markdown("### Engineered Feature: Origin Balance Error")
        if "errorBalanceOrig" in X.columns:
            fig_hist = px.histogram(
                X,
                x="errorBalanceOrig",
                nbins=50,
                title="Distribution of Origin Balance Errors",
                color_discrete_sequence=["#6366F1"],
            )
            st.plotly_chart(fig_hist, use_container_width=True)