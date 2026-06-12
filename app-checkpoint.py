import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(
    page_title="Credit Risk Command Center",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================= CSS =================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111f 0%, #0f172a 45%, #111827 100%);
    color: #f8fafc;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    display: none;
}

.block-container {
    padding-top: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

.navbar {
    background: rgba(15, 23, 42, 0.92);
    border: 1px solid #334155;
    border-radius: 18px;
    padding: 18px 28px;
    margin-bottom: 28px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.35);
}

.title {
    font-size: 34px;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 4px;
}

.subtitle {
    color: #94a3b8;
    font-size: 15px;
}

.card {
    background: rgba(30, 41, 59, 0.88);
    border: 1px solid #334155;
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.28);
}

.kpi-title {
    font-size: 13px;
    color: #94a3b8;
    margin-bottom: 8px;
}

.kpi-value {
    font-size: 30px;
    font-weight: 800;
    color: #f8fafc;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #e2e8f0;
    margin-top: 20px;
    margin-bottom: 14px;
}

.stRadio > div {
    background: rgba(30, 41, 59, 0.9);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 8px 14px;
}

.stRadio label {
    background: #0f172a;
    border-radius: 12px;
    padding: 10px 18px;
    margin-right: 8px;
    color: #cbd5e1 !important;
}

.stRadio label:hover {
    background: #1e40af;
}

.reason-card {
    background: rgba(15, 23, 42, 0.95);
    border-left: 5px solid #ef4444;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 12px;
}

div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD =================

@st.cache_data
def load_data():
    return pd.read_csv("./outputs/pd_results_final.csv")

@st.cache_resource
def load_model():
    return joblib.load("./outputs/final_xgb_model.pkl")

pd_results = load_data()
model = load_model()

lean_features = [
    "delinquency_score",
    "recent_delinquency",
    "payment_ratio",
    "avg_payment",
    "current_utilisation",
    "max_utilisation",
    "LIMIT_BAL"
]

# ================= HEADER =================

st.markdown("""
<div class="navbar">
    <div class="title">Credit Risk Command Center</div>
    <div class="subtitle">
        AI-powered Probability of Default prediction, portfolio risk monitoring,
        credit limit optimization, guardrails, and explainability.
    </div>
</div>
""", unsafe_allow_html=True)

page = st.radio(
    "",
    ["Executive Overview", "Customer Explorer", "New Customer Underwriting","Risk Analytics", "Explainability Center"],
    horizontal=True
)

def kpi(title, value):
    st.markdown(
        f"""
        <div class="card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def dark_fig(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        font_color="#e2e8f0",
        title_font_color="#f8fafc",
        legend_font_color="#e2e8f0"
    )
    return fig

# ================= EXECUTIVE =================

if page == "Executive Overview":

    st.markdown('<div class="section-title">Executive Portfolio Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi("Total Customers", f"{len(pd_results):,}")
    with c2:
        kpi("Average PD", f"{pd_results['predicted_pd'].mean():.2%}")
    with c3:
        kpi("High Risk Customers", f"{(pd_results['risk_segment'] == 'High Risk').sum():,}")
    with c4:
        kpi("Average Final Limit", f"₹{pd_results['final_recommended_limit'].mean():,.0f}")

    col1, col2 = st.columns(2)

    with col1:
        risk_counts = pd_results["risk_segment"].value_counts().reset_index()
        risk_counts.columns = ["Risk Segment", "Count"]

        fig = px.pie(
            risk_counts,
            names="Risk Segment",
            values="Count",
            hole=0.58,
            color="Risk Segment",
            color_discrete_map={
                "Low Risk": "#22c55e",
                "Medium Risk": "#f59e0b",
                "High Risk": "#ef4444"
            },
            title="Risk Segment Distribution"
        )
        st.plotly_chart(dark_fig(fig), use_container_width=True)

    with col2:
        cli_counts = pd_results["cli_decision"].value_counts().reset_index()
        cli_counts.columns = ["CLI Decision", "Count"]

        fig = px.pie(
            cli_counts,
            names="CLI Decision",
            values="Count",
            hole=0.58,
            color="CLI Decision",
            color_discrete_map={
                "Increase": "#22c55e",
                "Maintain": "#3b82f6",
                "Decrease": "#ef4444"
            },
            title="Credit Limit Decision Mix"
        )
        st.plotly_chart(dark_fig(fig), use_container_width=True)

    st.markdown('<div class="section-title">Portfolio Snapshot</div>', unsafe_allow_html=True)

    st.dataframe(
        pd_results[
            [
                "LIMIT_BAL",
                "predicted_pd",
                "risk_segment",
                "cli_decision",
                "final_recommended_limit"
            ]
        ].head(30),
        use_container_width=True
    )

# ================= CUSTOMER =================

elif page == "Customer Explorer":

    st.markdown('<div class="section-title">Customer Risk Explorer</div>', unsafe_allow_html=True)

    customer_index = st.selectbox("Select Customer", pd_results.index)
    customer = pd_results.loc[customer_index]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi("Predicted PD", f"{customer['predicted_pd']:.2%}")
    with c2:
        kpi("Risk Segment", customer["risk_segment"])
    with c3:
        kpi("CLI Decision", customer["cli_decision"])
    with c4:
        kpi("Final Limit", f"₹{customer['final_recommended_limit']:,.0f}")

    limit_df = pd.DataFrame({
        "Stage": ["Current Limit", "Recommended Limit", "Final Guardrailed Limit"],
        "Amount": [
            customer["LIMIT_BAL"],
            customer["recommended_limit"],
            customer["final_recommended_limit"]
        ]
    })

    fig = px.bar(
        limit_df,
        x="Stage",
        y="Amount",
        text="Amount",
        color="Stage",
        color_discrete_sequence=["#3b82f6", "#f59e0b", "#22c55e"],
        title="Credit Limit Movement"
    )
    fig.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
    st.plotly_chart(dark_fig(fig), use_container_width=True)

    st.markdown('<div class="section-title">Customer Feature Profile</div>', unsafe_allow_html=True)

    feature_df = customer[lean_features].reset_index()
    feature_df.columns = ["Feature", "Value"]

    fig = px.bar(
        feature_df,
        x="Value",
        y="Feature",
        orientation="h",
        color="Value",
        color_continuous_scale="Blues",
        title="Key Risk Features"
    )
    st.plotly_chart(dark_fig(fig), use_container_width=True)

# ================= ANALYTICS =================

elif page == "Risk Analytics":

    st.markdown('<div class="section-title">Risk Analytics</div>', unsafe_allow_html=True)

    fig = px.histogram(
        pd_results,
        x="predicted_pd",
        nbins=45,
        color="risk_segment",
        color_discrete_map={
            "Low Risk": "#22c55e",
            "Medium Risk": "#f59e0b",
            "High Risk": "#ef4444"
        },
        title="Predicted Probability of Default Distribution"
    )
    st.plotly_chart(dark_fig(fig), use_container_width=True)

    importance_df = pd.DataFrame({
        "Feature": lean_features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
        title="Final XGBoost Feature Importance"
    )
    st.plotly_chart(dark_fig(fig), use_container_width=True)

# ================= EXPLAINABILITY =================

elif page == "Explainability Center":

    st.markdown('<div class="section-title">SHAP Explainability Center</div>', unsafe_allow_html=True)

    customer_index = st.selectbox("Select Customer for Explanation", pd_results.index)

    X_customer = pd_results.loc[[customer_index], lean_features]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_customer)

    st.dataframe(X_customer, use_container_width=True)

    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_customer.iloc[0],
        feature_names=lean_features
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    shap.plots.waterfall(explanation, show=False)
    st.pyplot(fig)

    shap_reason_df = pd.DataFrame({
        "Feature": lean_features,
        "SHAP Value": shap_values[0]
    })

    shap_reason_df["Abs SHAP"] = shap_reason_df["SHAP Value"].abs()
    shap_reason_df = shap_reason_df.sort_values("Abs SHAP", ascending=False).head(3)

    st.markdown('<div class="section-title">Top Risk Drivers</div>', unsafe_allow_html=True)

    for _, row in shap_reason_df.iterrows():
        direction = "increased" if row["SHAP Value"] > 0 else "reduced"

        st.markdown(
            f"""
            <div class="reason-card">
                <b>{row['Feature']}</b><br>
                This feature {direction} the predicted default risk.
            </div>
            """,
            unsafe_allow_html=True
        )


#===============NEW CUSTOMER UNDERWRITING======================================================
elif page == "New Customer Underwriting":
    st.markdown(
    '<div class="section-title">New Customer Underwriting</div>',
    unsafe_allow_html=True
    )

    st.write(
    "Enter customer details to generate Probability of Default, Risk Segment and Credit Decision."
    )
    #input form
    col1,col2 = st.columns(2)

    with col1:

        LIMIT_BAL = st.number_input(
            "Credit Limit",
            min_value=10000,
            value=200000
        )

        delinquency_score = st.number_input(
            "Delinquency Score",
            min_value=0,
            value=0
        )

        recent_delinquency = st.number_input(
            "Recent Delinquency",
            min_value=0,
            value=0
        )

        payment_ratio = st.number_input(
            "Payment Ratio",
            min_value=0.0,
            value=0.80
        )

    with col2:

        avg_payment = st.number_input(
            "Average Payment",
            min_value=0,
            value=10000
        )

        current_utilisation = st.number_input(
            "Current Utilisation",
            min_value=0.0,
            value=0.30
        )

        max_utilisation = st.number_input(
            "Max Utilisation",
            min_value=0.0,
            value=0.50
        )
    #prediction button
    if st.button("Generate Decision"):
        customer = pd.DataFrame({

            "delinquency_score":[delinquency_score],

            "recent_delinquency":[recent_delinquency],

            "payment_ratio":[payment_ratio],

            "avg_payment":[avg_payment],

            "current_utilisation":[current_utilisation],

            "max_utilisation":[max_utilisation],

            "LIMIT_BAL":[LIMIT_BAL]
        })
        predicted_pd = model.predict_proba(
            customer
        )[:,1][0]
        #risk  decision
        def assign_risk_segment(pd):

            if pd < 0.10:
                return "Low Risk"

            elif pd < 0.30:
                return "Medium Risk"

            else:
                return "High Risk"

        risk_segment = assign_risk_segment(predicted_pd)
        #cli decision
        def cli_decision(pd_value,current_utilisation):

            if pd_value < 0.10 and current_utilisation < 0.50:
                return "Increase"

            elif pd_value >= 0.30:
                return "Decrease"

            else:
                return "Maintain"

        decision = cli_decision(
            predicted_pd,
            current_utilisation
        )
        #limit change %
        if decision == "Increase":

            limit_change_pct = 0.20

        elif decision == "Decrease":

            limit_change_pct = -0.20

        else:

            limit_change_pct = 0.0
        #recommended limit
        recommended_limit = (
            LIMIT_BAL *
            (1 + limit_change_pct)
        )
        #guardrails
        MAX_INCREASE = 0.25
        MAX_DECREASE = 0.20

        MIN_LIMIT = 10000
        MAX_LIMIT = 1000000

        upper_guardrail = (
            LIMIT_BAL *
            (1 + MAX_INCREASE)
        )

        lower_guardrail = (
            LIMIT_BAL *
            (1 - MAX_DECREASE)
        )

        guardrail_limit = min(
            max(
                recommended_limit,
                lower_guardrail
            ),
            upper_guardrail
        )

        final_recommended_limit = min(
            max(
                guardrail_limit,
                MIN_LIMIT
            ),
            MAX_LIMIT
        )
        #output display
        c1,c2,c3,c4 = st.columns(4)

        with c1:
            st.metric(
                "Probability of Default",
                f"{predicted_pd:.2%}"
            )

        with c2:
            st.metric(
                "Risk Segment",
                risk_segment
            )

        with c3:
            st.metric(
                "CLI Decision",
                decision
            )

        with c4:
            st.metric(
                "Recommended Limit",
                f"₹{final_recommended_limit:,.0f}"
            )
        #add buss summary
        st.subheader("Credit Decision Summary")

        summary = pd.DataFrame({
            "Metric":[
                "Probability of Default",
                "Risk Segment",
                "CLI Decision",
                "Current Limit",
                "Recommended Limit"
            ],

            "Value":[
                f"{predicted_pd:.2%}",
                risk_segment,
                decision,
                f"₹{LIMIT_BAL:,.0f}",
                f"₹{final_recommended_limit:,.0f}"
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True
        )