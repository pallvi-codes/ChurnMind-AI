# ==========================================
# ChurnMind AI
# Intelligent Customer Analytics Agent
# Part 1: Setup and Home Dashboard
# ==========================================


# Import Required Libraries

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib


# Page Configuration

st.set_page_config(
    page_title="ChurnMind AI",
    page_icon="📊",
    layout="wide"
)
st.markdown("""
<style>

.stApp{
    background: #EAF4FF;
}

h1,h2,h3{
    color:#1E3A8A !important;
    font-weight:700 !important;
}

[data-testid="stSidebar"]{
    background:#DCEEFF;
}

div[data-testid="metric-container"]{
    background:white;
    border-radius:18px;
    padding:18px;
    box-shadow:0 4px 15px rgba(0,0,0,.08);
}

.stButton>button{
    background:linear-gradient(90deg,#60A5FA,#3B82F6);
    color:white;
    border:none;
    border-radius:12px;
    padding:12px 20px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#2563EB;
    transform:scale(1.02);
}

div.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

</style>
""", unsafe_allow_html=True)

# Load Trained Model

model = joblib.load("churnmind_model.pkl")


# Load Feature Names

features = joblib.load("feature_names.pkl")


# Load Dataset

df = pd.read_csv(
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


# Sidebar Navigation

st.sidebar.title("ChurnMind AI")

menu = st.sidebar.radio(
    "Select Option",
    [
        "Home",
        "Dashboard",
        "Prediction",
        "Analytics",
        "AI Insights",
        "About Project"
    ]
)

# ==========================================
# Home Page
# ==========================================

if menu == "Home":

    # ------------------------------------------
    # Professional Header / Branding
    # ------------------------------------------

    st.title("🤖 ChurnMind AI")

    st.subheader("Intelligent Customer Analytics Agent")

    st.write(
        "AI-powered customer churn prediction, "
        "analytics and business intelligence platform."
    )

    st.divider()


    # ------------------------------------------
    # Welcome Section
    # ------------------------------------------

    st.subheader("👋 Welcome to ChurnMind AI")

    st.write(
        "ChurnMind AI is a machine learning based customer "
        "analytics platform designed to help businesses "
        "understand customer behaviour and identify customers "
        "who may be likely to churn."
    )


    # ------------------------------------------
    # Feature Cards
    # ------------------------------------------

    st.subheader("✨ What Can ChurnMind AI Do?")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            "📊 Customer Analytics\n\n"
            "Explore customer behaviour, churn patterns, "
            "retention trends and important business insights."
        )

    with col2:

        st.info(
            "🔮 Churn Prediction\n\n"
            "Use machine learning to estimate the probability "
            "of a customer leaving the service."
        )

    with col3:

        st.info(
            "🤖 AI Insights\n\n"
            "Generate useful insights and recommendations "
            "to support customer retention decisions."
        )


    st.divider()


    # ------------------------------------------
    # Advanced KPI Dashboard
    # ------------------------------------------

    st.subheader("📊 Customer Performance Overview")

    total_customers = len(df)

    churn_customers = len(
        df[df["Churn"] == "Yes"]
    )

    if total_customers > 0:

        churn_rate = (
            churn_customers / total_customers
        ) * 100

    else:

        churn_rate = 0


    average_monthly_charges = df["MonthlyCharges"].mean()

    average_tenure = df["tenure"].mean()


    # ------------------------------------------
    # KPI Cards
    # ------------------------------------------

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.metric(
            "👥 Total Customers",
            f"{total_customers:,}"
        )

    with col2:

        st.metric(
            "⚠️ Churned Customers",
            f"{churn_customers:,}"
        )

    with col3:

        st.metric(
            "📊 Churn Rate",
            f"{churn_rate:.2f}%"
        )

    with col4:

        st.metric(
            "💰 Avg Monthly Charges",
            f"${average_monthly_charges:.2f}"
        )

    with col5:

        st.metric(
            "📈 Avg Tenure",
            f"{average_tenure:.1f} months"
        )


    st.divider()

    # ------------------------------------------
    # Getting Started
    # ------------------------------------------

    st.subheader("🚀 Explore ChurnMind AI")

    st.write(
        "Use the sidebar to explore different capabilities "
        "of the ChurnMind AI platform."
    )

    st.success(
        "💡 Start with the Dashboard to explore customer "
        "churn patterns and analytics."
    )
# ==========================================
# Part 2: Dashboard Analytics
# ==========================================

elif menu == "Dashboard":

    st.header("Customer Dashboard")

    st.write(
        "Interactive overview of customer churn behaviour, "
        "customer charges and retention patterns."
    )

    # ------------------------------------------
    # Customer Overview
    # ------------------------------------------

    st.subheader("Customer Overview")

    total_customers = len(df)

    churn_customers = len(
        df[df["Churn"] == "Yes"]
    )

    retained_customers = len(
        df[df["Churn"] == "No"]
    )

    churn_rate = (
        churn_customers / total_customers
    ) * 100

    retention_rate = (
        retained_customers / total_customers
    ) * 100


    # Dashboard Cards

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Customers",
            total_customers
        )

    with col2:

        st.metric(
            "Churn Customers",
            churn_customers
        )

    with col3:

        st.metric(
            "Retained Customers",
            retained_customers
        )

    with col4:

        st.metric(
            "Churn Rate",
            f"{churn_rate:.2f}%"
        )


    st.divider()


    # ------------------------------------------
    # Customer Churn Distribution
    # ------------------------------------------

    st.subheader(
        "Customer Churn Distribution"
    )

    churn_count = df["Churn"].value_counts()

    fig1 = px.pie(
        values=churn_count.values,
        names=churn_count.index,
        title="Churn vs Retained Customers",
        hole=0.35
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )


    # ------------------------------------------
    # Contract Analysis
    # ------------------------------------------

    st.subheader(
        "Contract Wise Churn Analysis"
    )

    fig2 = px.histogram(
        df,
        x="Contract",
        color="Churn",
        barmode="group",
        title="Contract Type vs Churn"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )


    # ------------------------------------------
    # Monthly Charges Analysis
    # ------------------------------------------

    st.subheader(
        "Monthly Charges Analysis"
    )

    fig3 = px.box(
        df,
        x="Churn",
        y="MonthlyCharges",
        color="Churn",
        title="Monthly Charges Impact on Churn"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )


    # ------------------------------------------
    # Tenure Analysis
    # ------------------------------------------

    st.subheader(
        "Customer Tenure Analysis"
    )

    fig4 = px.histogram(
        df,
        x="tenure",
        color="Churn",
        title="Tenure Distribution"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )


    st.divider()


    # ==========================================
    # Additional Dashboard Analytics
    # ==========================================

    st.subheader(
        "Advanced Customer Analytics"
    )

    st.write(
        "Additional analysis of churn rates across "
        "contracts, internet services and payment methods."
    )


    # ------------------------------------------
    # Churn Rate by Contract
    # ------------------------------------------

    st.subheader(
        "Churn Rate by Contract Type"
    )

    contract_rate = (
        df.groupby("Contract")["Churn"]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
        .reset_index(name="ChurnRate")
    )

    fig5 = px.bar(
        contract_rate,
        x="Contract",
        y="ChurnRate",
        text="ChurnRate",
        title="Customer Churn Rate by Contract Type"
    )

    fig5.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )


    # ------------------------------------------
    # Churn Rate by Internet Service
    # ------------------------------------------

    st.subheader(
        "Churn Rate by Internet Service"
    )

    internet_rate = (
        df.groupby("InternetService")["Churn"]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
        .reset_index(name="ChurnRate")
    )

    fig6 = px.bar(
        internet_rate,
        x="InternetService",
        y="ChurnRate",
        text="ChurnRate",
        title="Customer Churn Rate by Internet Service"
    )

    fig6.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig6,
        use_container_width=True
    )


    # ------------------------------------------
    # Churn Rate by Payment Method
    # ------------------------------------------

    st.subheader(
        "Churn Rate by Payment Method"
    )

    payment_rate = (
        df.groupby("PaymentMethod")["Churn"]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
        .reset_index(name="ChurnRate")
    )

    fig7 = px.bar(
        payment_rate,
        x="PaymentMethod",
        y="ChurnRate",
        text="ChurnRate",
        title="Customer Churn Rate by Payment Method"
    )

    fig7.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig7,
        use_container_width=True
    )


    # ------------------------------------------
    # Customer Charges Summary
    # ------------------------------------------

    st.subheader(
        "Customer Charges Summary"
    )

    charge_col1, charge_col2, charge_col3 = st.columns(3)

    with charge_col1:

        st.metric(
            "Average Monthly Charges",
            f"${df['MonthlyCharges'].mean():.2f}"
        )

    with charge_col2:

        st.metric(
            "Maximum Monthly Charges",
            f"${df['MonthlyCharges'].max():.2f}"
        )

    with charge_col3:

        st.metric(
            "Minimum Monthly Charges",
            f"${df['MonthlyCharges'].min():.2f}"
        )


    # ------------------------------------------
    # Customer Tenure Summary
    # ------------------------------------------

    st.subheader(
        "Customer Tenure Summary"
    )

    tenure_col1, tenure_col2, tenure_col3 = st.columns(3)

    with tenure_col1:

        st.metric(
            "Average Tenure",
            f"{df['tenure'].mean():.1f} months"
        )

    with tenure_col2:

        st.metric(
            "Maximum Tenure",
            f"{df['tenure'].max()} months"
        )

    with tenure_col3:

        st.metric(
            "Minimum Tenure",
            f"{df['tenure'].min()} months"
        )


    # ------------------------------------------
    # Retention Overview
    # ------------------------------------------

    st.divider()

    st.subheader(
        "Retention Overview"
    )

    retention_col1, retention_col2 = st.columns(2)

    with retention_col1:

        st.metric(
            "Retention Rate",
            f"{retention_rate:.2f}%"
        )

    with retention_col2:

        st.metric(
            "Customers Retained",
            retained_customers
        )


    st.info(
        "Business Insight: Churn patterns can be analyzed "
        "using contract type, internet service, payment method, "
        "monthly charges and customer tenure."
    )


# ==========================================
# Part 3: Customer Churn Prediction
# ==========================================


elif menu == "Prediction":

    st.header("Customer Churn Prediction")

    st.write(
        "ChurnMind AI uses a trained machine learning model "
        "to estimate the probability of customer churn."
    )


    # ------------------------------------------
    # Prediction Information
    # ------------------------------------------

    st.subheader("Prediction Overview")

    st.info(
        "The prediction system analyzes the prepared customer "
        "feature set and generates a churn prediction with "
        "a probability-based risk assessment."
    )


    # ------------------------------------------
    # Customer Prediction Section
    # ------------------------------------------

    st.subheader("Customer Prediction")


    st.write(
        "Click the button below to run the trained churn "
        "prediction model."
    )


    # ------------------------------------------
    # Prepare Model Input
    # ------------------------------------------

    customer_data = {}

    for feature in features:

        customer_data[feature] = 0


    input_data = pd.DataFrame(
        [customer_data]
    )


    # Make sure feature order matches the trained model

    input_data = input_data[
        features
    ]


    # ------------------------------------------
    # Prediction Button
    # ------------------------------------------

    if st.button(
        "Predict Customer Churn",
        use_container_width=True
    ):


        # Model Prediction

        prediction = model.predict(
            input_data
        )


        # Churn Probability

        probability = model.predict_proba(
            input_data
        )[0][1]


        probability_percentage = (
            probability * 100
        )


        # --------------------------------------
        # Prediction Result
        # --------------------------------------

        st.divider()

        st.subheader(
            "Prediction Result"
        )


        if prediction[0] == 1:

            result = "Customer is likely to churn"

            st.error(result)

        else:

            result = "Customer is likely to stay"

            st.success(result)


        # --------------------------------------
        # Probability
        # --------------------------------------

        st.subheader(
            "Churn Probability"
        )


        st.metric(
            "Estimated Churn Probability",
            f"{probability_percentage:.2f}%"
        )


        st.progress(
            float(probability)
        )


        # --------------------------------------
        # Risk Classification
        # --------------------------------------

        st.subheader(
            "Customer Risk Level"
        )


        if probability >= 0.70:

            risk = "High Risk"

            risk_description = (
                "The customer has a high estimated probability "
                "of churn and should be prioritized for retention."
            )

            recommendation = (
                "Provide personalized offers, proactive customer "
                "support, loyalty benefits and targeted retention campaigns."
            )

            st.error(
                f"Risk Level: {risk}"
            )


        elif probability >= 0.40:

            risk = "Medium Risk"

            risk_description = (
                "The customer shows a moderate probability of churn "
                "and should be monitored regularly."
            )

            recommendation = (
                "Improve customer engagement, monitor service usage "
                "and consider personalized offers."
            )

            st.warning(
                f"Risk Level: {risk}"
            )


        else:

            risk = "Low Risk"

            risk_description = (
                "The customer has a relatively low estimated "
                "probability of churn."
            )

            recommendation = (
                "Maintain service quality and continue regular "
                "customer engagement."
            )

            st.success(
                f"Risk Level: {risk}"
            )


        # --------------------------------------
        # Risk Explanation
        # --------------------------------------

        st.subheader(
            "Risk Assessment"
        )

        st.write(
            risk_description
        )


        # --------------------------------------
        # AI Recommendation
        # --------------------------------------

        st.subheader(
            "AI Retention Recommendation"
        )

        st.info(
            recommendation
        )


        # --------------------------------------
        # Prediction Summary
        # --------------------------------------

        st.divider()

        st.subheader(
            "Prediction Summary"
        )


        summary_col1, summary_col2, summary_col3 = st.columns(3)


        with summary_col1:

            st.metric(
                "Prediction",
                (
                    "Churn"
                    if prediction[0] == 1
                    else "Retained"
                )
            )


        with summary_col2:

            st.metric(
                "Probability",
                f"{probability_percentage:.2f}%"
            )


        with summary_col3:

            st.metric(
                "Risk Level",
                risk
            )


        # --------------------------------------
        # Business Action
        # --------------------------------------

        st.subheader(
            "Recommended Business Action"
        )

        if risk == "High Risk":

            st.write(
                "1. Contact the customer proactively."
            )

            st.write(
                "2. Provide a personalized retention offer."
            )

            st.write(
                "3. Review service quality and customer complaints."
            )

            st.write(
                "4. Monitor the customer after intervention."
            )


        elif risk == "Medium Risk":

            st.write(
                "1. Monitor customer engagement."
            )

            st.write(
                "2. Provide relevant service recommendations."
            )

            st.write(
                "3. Consider loyalty or promotional offers."
            )


        else:

            st.write(
                "1. Maintain service quality."
            )

            st.write(
                "2. Continue regular customer engagement."
            )

            st.write(
                "3. Encourage long-term customer loyalty."
            )


        # --------------------------------------
        # Model Input Information
        # --------------------------------------

        with st.expander(
            "View Model Input Features"
        ):

            st.write(
                "The trained model uses the following "
                "prepared features:"
            )

            st.write(
                features
            )

# ==========================================
# Part 4: Advanced Customer Analytics
# ==========================================

elif menu == "Analytics":

    st.header("Customer Analytics")

    st.write(
        "Detailed analysis of customer behaviour, "
        "churn patterns and retention indicators."
    )

    # ------------------------------------------
    # Key Metrics
    # ------------------------------------------

    st.subheader("Key Analytics Metrics")

    total_customers = len(df)

    churn_customers = len(
        df[df["Churn"] == "Yes"]
    )

    retained_customers = len(
        df[df["Churn"] == "No"]
    )

    churn_percentage = (
        churn_customers / total_customers
    ) * 100

    retention_percentage = (
        retained_customers / total_customers
    ) * 100

    average_monthly_charges = (
        df["MonthlyCharges"].mean()
    )

    average_tenure = (
        df["tenure"].mean()
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Churn Percentage",
            f"{churn_percentage:.2f}%"
        )

    with col2:
        st.metric(
            "Retention Percentage",
            f"{retention_percentage:.2f}%"
        )

    with col3:
        st.metric(
            "Average Tenure",
            f"{average_tenure:.1f} months"
        )

    st.divider()

    # ------------------------------------------
    # Gender Analysis
    # ------------------------------------------

    st.subheader("Gender Wise Churn Analysis")

    gender_data = (
        df.groupby(["gender", "Churn"])
        .size()
        .reset_index(name="Customers")
    )

    fig1 = px.bar(
        gender_data,
        x="gender",
        y="Customers",
        color="Churn",
        barmode="group",
        title="Gender vs Customer Churn"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # ------------------------------------------
    # Internet Service Analysis
    # ------------------------------------------

    st.subheader("Internet Service Analysis")

    internet_data = (
        df.groupby(["InternetService", "Churn"])
        .size()
        .reset_index(name="Customers")
    )

    fig2 = px.bar(
        internet_data,
        x="InternetService",
        y="Customers",
        color="Churn",
        barmode="group",
        title="Internet Service vs Churn"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # ------------------------------------------
    # Payment Method Analysis
    # ------------------------------------------

    st.subheader("Payment Method Analysis")

    payment_data = (
        df.groupby(["PaymentMethod", "Churn"])
        .size()
        .reset_index(name="Customers")
    )

    fig3 = px.bar(
        payment_data,
        x="PaymentMethod",
        y="Customers",
        color="Churn",
        barmode="group",
        title="Payment Method vs Churn"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    # ------------------------------------------
    # Senior Citizen Analysis
    # ------------------------------------------

    st.subheader("Senior Citizen Analysis")

    senior_data = (
        df.groupby(["SeniorCitizen", "Churn"])
        .size()
        .reset_index(name="Customers")
    )

    senior_data["SeniorCitizen"] = (
        senior_data["SeniorCitizen"]
        .map({
            0: "Non-Senior Citizen",
            1: "Senior Citizen"
        })
    )

    fig4 = px.bar(
        senior_data,
        x="SeniorCitizen",
        y="Customers",
        color="Churn",
        barmode="group",
        title="Senior Citizen vs Churn"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

    # ------------------------------------------
    # Monthly Charges Distribution
    # ------------------------------------------

    st.subheader("Monthly Charges Distribution")

    fig5 = px.histogram(
        df,
        x="MonthlyCharges",
        color="Churn",
        nbins=30,
        title="Monthly Charges Distribution by Churn"
    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )

    # ------------------------------------------
    # Tenure vs Monthly Charges
    # ------------------------------------------

    st.subheader("Tenure vs Monthly Charges")

    fig6 = px.scatter(
        df,
        x="tenure",
        y="MonthlyCharges",
        color="Churn",
        title="Tenure vs Monthly Charges",
        hover_data=[
            "Contract",
            "InternetService",
            "PaymentMethod"
        ]
    )

    st.plotly_chart(
        fig6,
        use_container_width=True
    )

    # ------------------------------------------
    # Churn Rate by Contract
    # ------------------------------------------

    st.subheader("Churn Rate by Contract")

    contract_rate = (
        df.groupby("Contract")["Churn"]
        .apply(
            lambda x: (x == "Yes").mean() * 100
        )
        .reset_index(name="ChurnRate")
    )

    fig7 = px.bar(
        contract_rate,
        x="Contract",
        y="ChurnRate",
        text="ChurnRate",
        title="Churn Rate by Contract Type"
    )

    fig7.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig7,
        use_container_width=True
    )

    # ------------------------------------------
    # Statistical Summary
    # ------------------------------------------

    st.subheader("Customer Statistical Summary")

    summary = pd.DataFrame({
        "Metric": [
            "Total Customers",
            "Churn Customers",
            "Retained Customers",
            "Churn Percentage",
            "Retention Percentage",
            "Average Monthly Charges",
            "Average Tenure"
        ],
        "Value": [
            total_customers,
            churn_customers,
            retained_customers,
            f"{churn_percentage:.2f}%",
            f"{retention_percentage:.2f}%",
            f"${average_monthly_charges:.2f}",
            f"{average_tenure:.1f} months"
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    # ------------------------------------------
    # Business Insights
    # ------------------------------------------

    st.divider()

    st.subheader("Business Insights")

    st.info(
        "Customer churn can be analyzed using contract type, "
        "internet service, payment method, monthly charges and tenure."
    )

    st.info(
        "Customers with shorter tenure may require stronger "
        "engagement and retention strategies."
    )

    st.info(
        "Businesses can use these analytics to identify "
        "high-risk customer segments and design targeted "
        "retention campaigns."
    )

    st.success(
        "Combining customer analytics with machine learning "
        "predictions can support data-driven retention decisions."
    )


# ==========================================
# Part 5: AI Insights and Chat Assistant
# ==========================================

elif menu == "AI Insights":

    st.header("AI Insights and Chat Assistant")

    st.write(
        "ChurnMind AI provides customer churn insights, "
        "retention recommendations and project-related answers."
    )

    # ------------------------------------------
    # Customer Churn Insights
    # ------------------------------------------

    st.subheader("Customer Churn Insights")

    total_customers = len(df)

    churn_customers = len(
        df[df["Churn"] == "Yes"]
    )

    retained_customers = len(
        df[df["Churn"] == "No"]
    )

    churn_rate = (
        churn_customers / total_customers
    ) * 100

    retention_rate = (
        retained_customers / total_customers
    ) * 100

    st.metric(
        "Current Churn Rate",
        f"{churn_rate:.2f}%"
    )

    st.metric(
        "Current Retention Rate",
        f"{retention_rate:.2f}%"
    )

    st.divider()

    # ------------------------------------------
    # Automatic Business Insights
    # ------------------------------------------

    st.subheader("Automatic Business Insights")

    contract_churn = (
        df.groupby("Contract")["Churn"]
        .apply(
            lambda x: (x == "Yes").mean() * 100
        )
    )

    highest_contract = contract_churn.idxmax()

    highest_contract_rate = contract_churn.max()

    internet_churn = (
        df.groupby("InternetService")["Churn"]
        .apply(
            lambda x: (x == "Yes").mean() * 100
        )
    )

    highest_internet = internet_churn.idxmax()

    highest_internet_rate = internet_churn.max()

    st.info(
        f"The contract category with the highest observed "
        f"churn rate is '{highest_contract}' with approximately "
        f"{highest_contract_rate:.2f}% churn."
    )

    st.info(
        f"The internet service category with the highest "
        f"observed churn rate is '{highest_internet}' with "
        f"approximately {highest_internet_rate:.2f}% churn."
    )

    st.info(
        f"The average monthly charge across customers is "
        f"${df['MonthlyCharges'].mean():.2f}."
    )

    st.info(
        f"The average customer tenure is "
        f"{df['tenure'].mean():.1f} months."
    )

    st.divider()

    # ------------------------------------------
    # Retention Strategies
    # ------------------------------------------

    st.subheader("Customer Retention Strategies")

    strategies = [
        "Identify high-risk customers before they leave.",
        "Provide personalized offers to high-risk customers.",
        "Encourage customers to choose suitable long-term contracts.",
        "Improve customer support and service experience.",
        "Monitor customer behaviour and engagement regularly.",
        "Use customer feedback to improve services.",
        "Develop loyalty programs for long-term customers."
    ]

    for strategy in strategies:

        st.write(
            "• " + strategy
        )

    st.divider()

        # ------------------------------------------
    # Ask ChurnMind AI
    # ------------------------------------------

    st.subheader("Ask ChurnMind AI")

    st.write(
        "Ask questions about customer churn, the dataset, "
        "machine learning, project workflow, analytics or "
        "customer retention."
    )

    question = st.text_input(
        "Ask your customer analytics question",
        placeholder="Example: What is the churn rate?"
    )

    ask_button = st.button(
        "Send",
        use_container_width=True
    )

    if ask_button:

        if question.strip():

            question = question.lower().strip()

            # --------------------------------------
            # Question Categories
            # --------------------------------------

            answers = {

            "what is churnmind ai":
                "ChurnMind AI is an intelligent customer analytics "
                "agent that uses machine learning to predict customer "
                "churn and provide business insights.",

            "what is customer churn":
                "Customer churn occurs when a customer stops using "
                "a company's product or service.",

            "why churn prediction is important":
                "Churn prediction helps businesses identify customers "
                "who may leave and take preventive retention actions.",

            "what is the objective":
                "The objective is to analyze customer data, predict "
                "churn and provide useful customer retention insights.",

            "what is the objective of this project":
                "The objective is to analyze customer data, predict "
                "churn and provide useful customer retention insights.",

            "how does churnmind ai work":
                "The system loads customer data, performs analysis, "
                "uses a trained machine learning model for prediction "
                "and presents results through Streamlit.",

            "explain the workflow":
                "The workflow includes data collection, preprocessing, "
                "exploratory data analysis, model training, prediction, "
                "evaluation and Streamlit deployment.",

            "which machine learning algorithm":
                "Logistic Regression is used for the churn prediction "
                "task because churn is a binary classification problem.",

            "why logistic regression":
                "Logistic Regression is suitable for binary classification "
                "and can provide probability-based predictions.",

            "what is logistic regression":
                "Logistic Regression is a supervised machine learning "
                "algorithm commonly used for binary classification.",

            "what is machine learning":
                "Machine Learning allows computer systems to learn "
                "patterns from data and make predictions.",

            "what is artificial intelligence":
                "Artificial Intelligence refers to techniques that enable "
                "computer systems to perform tasks involving learning, "
                "reasoning and decision-making.",

            "what is supervised learning":
                "Supervised learning trains a model using labeled examples "
                "where the expected output is already known.",

            "what is classification":
                "Classification is a supervised learning task where a model "
                "predicts a category or class.",

            "what is the dataset":
                "The project uses a telecom customer churn dataset containing "
                "customer demographics, services, charges, tenure and churn status.",

            "what columns are in the dataset":
                "The dataset contains customer information such as gender, "
                "SeniorCitizen, Partner, Dependents, tenure, services, "
                "Contract, PaymentMethod, MonthlyCharges, TotalCharges and Churn.",

            "what is the target variable":
                "The target variable is Churn. It indicates whether a customer "
                "left the service or remained with the company.",

            "what are features":
                "Features are input variables used by the machine learning "
                "model to make predictions.",

            "what is data preprocessing":
                "Data preprocessing prepares raw data for machine learning "
                "by cleaning, transforming and organizing the data.",

            "why data cleaning":
                "Data cleaning improves data quality by handling missing values, "
                "duplicates, incorrect data types and inconsistencies.",

            "what is exploratory data analysis":
                "Exploratory Data Analysis identifies patterns, trends and "
                "relationships using statistics and visualizations.",

            "what is pandas":
                "Pandas is a Python library used for data manipulation, "
                "cleaning and analysis.",

            "what is numpy":
                "NumPy is a Python library designed for numerical computing "
                "and array-based operations.",

            "what is matplotlib":
                "Matplotlib is a Python visualization library used to create "
                "graphs and charts.",

            "what is seaborn":
                "Seaborn is a Python visualization library built on Matplotlib "
                "and commonly used for statistical visualizations.",

            "what is plotly":
                "Plotly is an interactive visualization library used to create "
                "interactive charts and graphs.",

            "what is streamlit":
                "Streamlit is a Python framework that makes it easy to build "
                "interactive data science and machine learning web applications.",

            "why streamlit":
                "Streamlit provides a simple way to convert Python-based "
                "data science projects into interactive web applications.",

            "what is confusion matrix":
                "A confusion matrix compares actual and predicted classes "
                "and helps evaluate classification performance.",

            "what is accuracy":
                "Accuracy is the proportion of correct predictions among "
                "all predictions.",

            "what is precision":
                "Precision measures how many observations predicted as positive "
                "were actually positive.",

            "what is recall":
                "Recall measures how many actual positive observations were "
                "correctly identified by the model.",

            "how contract affects churn":
                "Contract type can strongly influence churn behaviour. "
                "Different contract categories may have different observed "
                "churn rates.",

            "how tenure affects churn":
                "Tenure represents the length of the customer relationship. "
                "It can be useful for identifying different customer behaviour "
                "patterns.",

            "how monthly charges affect churn":
                "Monthly charges can be associated with customer churn behaviour. "
                "The dashboard allows this relationship to be explored visually.",

            "how payment method affects churn":
                "Payment method can be analyzed to identify differences in "
                "churn behaviour across payment categories.",

            "how internet service affects churn":
                "Different internet service categories can show different "
                "observed churn rates in the dataset.",

            "how to reduce churn":
                "Businesses can reduce churn by identifying high-risk customers, "
                "improving support, offering personalized retention campaigns "
                "and improving service quality.",

            "what is customer retention":
                "Customer retention means keeping existing customers and "
                "maintaining long-term customer relationships.",

            "what is dashboard":
                "A dashboard is an interactive interface that presents "
                "important metrics, charts and business insights.",

            "what graphs are used":
                "The project uses pie charts, bar charts, histograms, box plots, "
                "scatter plots and correlation analysis.",

            "what is business analytics":
                "Business analytics uses data, statistics and analytical "
                "techniques to support better business decisions.",

            "what problem does this project solve":
                "The project helps businesses identify customers who may churn "
                "and supports data-driven customer retention decisions.",

            "who can use this project":
                "Telecom companies and other subscription-based businesses "
                "can use churn analytics to improve customer retention.",

            "what are project advantages":
                "Advantages include automated churn prediction, interactive "
                "analytics, customer risk assessment and business insights.",

            "what are limitations":
                "The project depends on the quality and relevance of the "
                "available customer dataset. Predictions may change when "
                "customer behaviour or data patterns change.",

            "what is future scope":
                "Future improvements could include real-time data integration, "
                "advanced machine learning models, automated alerts and a more "
                "advanced conversational AI system.",

            "explain your project":
                "ChurnMind AI is a machine learning-based customer analytics "
                "application that analyzes telecom customer data, predicts "
                "churn and provides retention-oriented business insights.",

            "what technologies are used":
                "The project uses Python, Pandas, NumPy, Matplotlib, Seaborn, "
                "Plotly, Scikit-learn, Joblib and Streamlit.",

            "what is model training":
                "Model training is the process of allowing a machine learning "
                "algorithm to learn patterns from training data.",

            "what is model evaluation":
                "Model evaluation measures how well a trained model performs "
                "using suitable evaluation metrics.",

            "what is prediction":
                "Prediction is the process of using a trained model to estimate "
                "the outcome for given input data.",

            "what is risk level":
                "Risk level represents the estimated likelihood of customer churn. "
                "In this application it is categorized as Low, Medium or High risk.",

            "what recommendation does the system provide":
                "The system provides retention-oriented recommendations such as "
                "customer engagement, personalized offers and proactive support.",

            "what is customer analytics":
                "Customer analytics involves analyzing customer data to understand "
                "behaviour, preferences, engagement and churn patterns.",

            "which contract has the most customers":
                f"The contract with the largest number of customers is "
                f"{df['Contract'].value_counts().idxmax()}.",

            "which payment method is most common":
                f"The most common payment method is "
                f"{df['PaymentMethod'].value_counts().idxmax()}.",

            "which internet service is most common":
                f"The most common internet service is "
                f"{df['InternetService'].value_counts().idxmax()}.",

            "what is the highest monthly charge":
                f"The highest monthly charge in the dataset is "
                f"${df['MonthlyCharges'].max():.2f}.",

            "what is the lowest monthly charge":
                f"The lowest monthly charge in the dataset is "
                f"${df['MonthlyCharges'].min():.2f}.",

            "what is the maximum tenure":
                f"The maximum customer tenure is "
                f"{df['tenure'].max()} months.",

            "what is the minimum tenure":
                f"The minimum customer tenure is "
                f"{df['tenure'].min()} months.",

            "how many senior citizens":
                f"There are "
                f"{df['SeniorCitizen'].sum():,} senior citizen customers "
                f"in the dataset.",

            "how many customers use fiber optic":
                f"There are "
                f"{(df['InternetService'] == 'Fiber optic').sum():,} "
                f"customers using Fiber optic internet service.",

            "how many customers use dsl":
                f"There are "
                f"{(df['InternetService'] == 'DSL').sum():,} "
                f"customers using DSL internet service.",

            "which gender has more customers":
                f"The gender with more customers is "
                f"{df['gender'].value_counts().idxmax()}."
        }


        # --------------------------------------
        # Flexible Question Matching
        # --------------------------------------

        answer_found = False

        for key, answer in answers.items():

            if key in question:

                st.success(answer)

                answer_found = True

                break


        # --------------------------------------
        # Additional Keyword Matching
        # --------------------------------------

        if not answer_found:

            if "churn" in question and "rate" in question:

                st.success(
                    f"The current churn rate in the dataset is "
                    f"{churn_rate:.2f}%."
                )

                answer_found = True


            elif "retention" in question:

                st.success(
                    f"The current customer retention rate is "
                    f"{retention_rate:.2f}%."
                )

                answer_found = True


            elif "average" in question and "tenure" in question:

                st.success(
                    f"The average customer tenure is "
                    f"{df['tenure'].mean():.1f} months."
                )

                answer_found = True


            elif "average" in question and "monthly" in question:

                st.success(
                    f"The average monthly charge is "
                    f"${df['MonthlyCharges'].mean():.2f}."
                )

                answer_found = True


        # --------------------------------------
        # Unknown Question
        # --------------------------------------

        if not answer_found:

            st.warning(
                "I can answer questions about ChurnMind AI, "
                "customer churn, the dataset, machine learning, "
                "analytics, Streamlit, the project workflow and "
                "customer retention."
            )
    

# ==========================================
# Part 6: About Project and Final Information
# ==========================================

elif menu == "About Project":

    st.header("About ChurnMind AI")

    # ------------------------------------------
    # Developer Name
    # ------------------------------------------

    st.markdown(
        """
        <div style="text-align:center; margin-bottom:25px;">
            <p style="font-size:17px; margin-bottom:5px;">
                Developed by
            </p>
            <h2 style="font-family:cursive; margin-top:0;">
                ✦ Pallvi ✦
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        "ChurnMind AI is an intelligent customer analytics agent "
        "developed to analyze customer churn behaviour, provide "
        "machine learning-based predictions, generate business "
        "insights and support data-driven customer retention decisions."
    )

    st.divider()

    # ------------------------------------------
    # Project Overview
    # ------------------------------------------

    st.subheader("Project Overview")

    st.write(
        """
        ChurnMind AI combines customer data analysis,
        machine learning, interactive visualization and
        an AI-powered question-answering assistant to
        understand customer churn behaviour.

        The application analyzes customer characteristics,
        services, tenure, charges and other factors and
        presents meaningful business insights through an
        interactive Streamlit dashboard.

        The AI assistant allows users to ask questions about
        customer churn, the dataset, machine learning,
        project workflow, analytics and customer retention.
        """
    )

    # ------------------------------------------
    # Project Objective
    # ------------------------------------------

    st.subheader("Project Objective")

    objectives = [
        "Analyze customer churn behaviour.",
        "Identify important customer churn patterns.",
        "Use machine learning for churn prediction.",
        "Estimate customer churn probability.",
        "Classify customers according to risk level.",
        "Provide customer retention recommendations.",
        "Provide AI-powered project and analytics assistance.",
        "Present analytical results through an interactive dashboard."
    ]

    for objective in objectives:

        st.write(
            "• " + objective
        )

    # ------------------------------------------
    # Technologies Used
    # ------------------------------------------

    st.subheader("Technologies Used")

    technologies = [
        "Python",
        "Pandas",
        "NumPy",
        "Matplotlib",
        "Seaborn",
        "Plotly",
        "Scikit-learn",
        "Joblib",
        "Streamlit"
    ]

    for technology in technologies:

        st.write(
            "• " + technology
        )

    # ------------------------------------------
    # Machine Learning
    # ------------------------------------------

    st.subheader("Machine Learning")

    st.write(
        """
        The project uses a trained machine learning model
        to perform customer churn prediction.

        Logistic Regression is used as the primary
        classification approach for predicting whether
        a customer is likely to churn or stay.

        The model uses customer-related features to identify
        patterns associated with customer churn and supports
        customer risk assessment.
        """
    )

    # ------------------------------------------
    # AI Assistant
    # ------------------------------------------

    st.subheader("AI Assistant")

    st.write(
        """
        ChurnMind AI includes an interactive question-answering
        assistant that allows users to ask questions related to
        customer churn, machine learning, data analysis,
        project workflow and customer retention.

        Users can enter a question and click the Send button
        to receive a relevant response from the assistant.
        """
    )

    # ------------------------------------------
    # Main Features
    # ------------------------------------------

    st.subheader("Main Project Features")

    project_features = [
        "Interactive Customer Dashboard",
        "Customer Churn Analysis",
        "Customer Churn Prediction",
        "Churn Probability Estimation",
        "Customer Risk Classification",
        "Interactive Data Visualizations",
        "AI-powered Customer Analytics Agent",
        "AI Business Insights",
        "Customer Retention Recommendations",
        "Interactive Question and Answer Assistant",
        "Statistical Customer Analysis"
    ]

    for item in project_features:

        st.write(
            "✓ " + item
        )

    # ------------------------------------------
    # Business Value
    # ------------------------------------------

    st.subheader("Business Value")

    st.write(
        """
        ChurnMind AI can help businesses understand
        customer behaviour and identify potential churn
        patterns.

        These insights can support proactive customer
        engagement, targeted retention strategies and
        data-driven decision making.

        By identifying customers who may be at higher risk
        of churn, businesses can take preventive actions
        and improve customer retention.
        """
    )

    # ------------------------------------------
    # Future Scope
    # ------------------------------------------

    st.subheader("Future Scope")

    future_scope = [
        "Real-time customer churn prediction.",
        "Integration with live business databases.",
        "Advanced machine learning and deep learning models.",
        "Automated high-risk customer alerts.",
        "Advanced conversational AI integration.",
        "Personalized retention campaign generation.",
        "Deployment on cloud infrastructure."
    ]

    for item in future_scope:

        st.write(
            "• " + item
        )

    # ------------------------------------------
    # Project Summary
    # ------------------------------------------

    st.divider()

    st.subheader("Project Summary")

    st.success(
        "ChurnMind AI is an intelligent customer analytics agent "
        "developed by Pallvi. It combines customer data analysis, "
        "machine learning, interactive visualization and an AI "
        "question-answering assistant to analyze customer churn, "
        "predict customer risk and support data-driven customer "
        "retention decisions."
    )


# ==========================================
# Report Download Section
# ==========================================

st.sidebar.divider()

st.sidebar.subheader(
    "Project Resources"
)

report_data = pd.DataFrame(
    {
        "Project": [
            "ChurnMind AI"
        ],

        "Purpose": [
            "Customer Churn Prediction and Analytics"
        ],

        "Technology": [
            "Python, Pandas, NumPy, Scikit-learn, Streamlit"
        ],

        "Model": [
            "Logistic Regression"
        ],

        "Application": [
            "Customer Analytics and Retention"
        ],

        "Developer": [
            "Pallvi"
        ]
    }
)

csv = report_data.to_csv(
    index=False
)

st.sidebar.download_button(
    label="Download Project Summary",
    data=csv,
    file_name="ChurnMind_AI_Project_Summary.csv",
    mime="text/csv"
)


# ==========================================
# Application Footer
# ==========================================

st.sidebar.divider()

st.sidebar.write(
    "ChurnMind AI"
)

st.sidebar.caption(
    "Intelligent Customer Analytics Agent"
)

st.sidebar.markdown(
    """
    <p style="font-family:cursive; font-size:16px;">
        ✦ Developer: Pallvi ✦
    </p>
    """,
    unsafe_allow_html=True
)